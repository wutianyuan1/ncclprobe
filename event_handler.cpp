#include "event_handler.hpp"
#include "comm.hpp"
#include "utils.hpp"


static void parse_task(std::string task, int* role, int* target_peer)
{
    size_t pos = task.find('_');
    if (pos != std::string::npos) {
        *role = std::stoi(task.substr(0, pos));
        *target_peer = std::stoi(task.substr(pos + 1));
    }
}


ProfileResult p2p_profile_task(int role, int peer, ncclComm_t comm = nullptr, cudaStream_t stream = nullptr)
{
    // int* buf;
    // cudaMalloc(&buf, 1024 * 1024);
    
    if (role == (int)ProcessRole::ROLE_SENDER) {
        printf("[Rank %d] I am sender, my target is %d\n", get_rank(DistEngine::auto_find), peer);
    } else if (role == (int)ProcessRole::ROLE_RECVER) {
        // ncclRecv(buf, 1024 * 1024, ncclInt, peer, comm?, stream?)
        printf("[Rank %d] I am recver, my sender is %d\n", get_rank(DistEngine::auto_find), peer);
    }
    return ProfileResult(1, 2, 3);
}


ProfileResult::ProfileResult(double minl, double maxl, double avgl)
    : min_lat(minl), max_lat(maxl), avg_lat(avgl) 
{};

std::string ProfileResult::serialize()
{
    std::stringstream ss;
    ss.write(reinterpret_cast<const char*>(&min_lat), sizeof(min_lat));
    ss.write(reinterpret_cast<const char*>(&max_lat), sizeof(max_lat));
    ss.write(reinterpret_cast<const char*>(&avg_lat), sizeof(avg_lat));
    return ss.str();
}


EventHandler::EventHandler(std::string master_addr, int port)
{
    world_comm = nullptr;
    client = std::shared_ptr<cpp_redis::client>(new cpp_redis::client());
    client->connect(master_addr, port);
    cudaMalloc(&this->control_state, sizeof(int) * 4);
}

EventHandler::~EventHandler()
{
    cudaFree(this->control_state);
    client->disconnect();
}

bool EventHandler::has_world_comm() const
{
    return this->world_comm != nullptr;
}

void EventHandler::set_world_comm(ncclComm_t comm)
{
    this->world_comm = comm;
}

void EventHandler::fetech_and_exec_task()
{
    // check my task to do in this term.
    std::string task_name = std::string("validtask_rank_") + std::to_string(get_rank(DistEngine::auto_find));
    std::string task_content;
    client->get(task_name,
        [&](const cpp_redis::reply& reply) {
            if (reply.is_string())
                task_content = reply.as_string();
        }
    );
    client->sync_commit();

    // If a task is already acked by the worker, just skip it.
    if (task_content.size() >= 10 && task_content == std::string(TASK_ACKED))
        return;
    // Otherwise, receive this task and prepare to execute it.
    client->set(task_name, TASK_ACKED);
    client->sync_commit();

    int role, peer;
    parse_task(task_content, &role, &peer);
    // Perform p2p send/recv job and get exec time metrics
    // FIXME: pass correct comm and stream
    auto result = p2p_profile_task(role, peer, nullptr, nullptr);
    // Add the result to redis
    client->set(task_name + std::string("_result"), result.serialize());
    client->sync_commit();
}

void EventHandler::handle_control_signal(cudaStream_t curr_stream)
{
    Communicator parsed_comm;
    parse_communicator(world_comm, &parsed_comm);

    if (parsed_comm.group_rank == 0)
    {
        int my_pause = 0;
        client->get("control_state",
            [&](const cpp_redis::reply& reply) {
                if (reply.is_string() && reply.as_string()[0] == '2') my_pause = 1;
            }
        );
        client->sync_commit();
        if (my_pause)
            cudaMemcpy(control_state, &my_pause, sizeof(int), cudaMemcpyHostToDevice);
        ncclBroadcast(control_state, control_state, 1, ncclInt, 0, world_comm, curr_stream);
    } else {
        ncclBroadcast(nullptr, control_state, 1, ncclInt, 0, world_comm, curr_stream);
    }
    int sync_pause = 0;
    cudaMemcpy(&sync_pause, control_state, sizeof(int), cudaMemcpyDeviceToHost);
    if (sync_pause)
        BOOST_LOG_TRIVIAL(info) << "Rank " << get_rank(DistEngine::auto_find) << " receives pause signal, paused!";

    // If we should pause, then loop to wait the "continue" (pause=0) signal 
    if (sync_pause == 1)
    {
        while (true)
        {
            // fetch a task from redis and execute it
            fetech_and_exec_task();
        
            // check if we should continue
            if (parsed_comm.group_rank == 0)
            {
                int my_pause = 1;
                client->get("control_state",
                    [&](const cpp_redis::reply& reply) {
                        if (reply.is_string() && reply.as_string()[0] != '2') my_pause = 0;
                    }
                );
                client->sync_commit();
                if (my_pause == 0)
                    cudaMemcpy(control_state, &my_pause, sizeof(int), cudaMemcpyHostToDevice);
                ncclBroadcast(control_state, control_state, 1, ncclInt, 0, world_comm, curr_stream);
            } else {
                ncclBroadcast(nullptr, control_state, 1, ncclInt, 0, world_comm, curr_stream);
            }
            int should_block = 0;
            cudaMemcpy(&should_block, control_state, sizeof(int), cudaMemcpyDeviceToHost);
            if (!should_block)
            {
                BOOST_LOG_TRIVIAL(info) << "Rank " << get_rank(DistEngine::auto_find) << " receives continue signal, will continue!";
                break;   
            }
            usleep(500 * 1000);  // wait for 0.5 seconds
        }
    }
}
