import argparse
import time
import logging
import redis
import numpy as np
import redis.client
import local_analyzer

PROFILE_PERIOD = 20


class LocalController(object):
    def __init__(self, redis_ip: str, redis_port: int, node_ip: str,
                 config_path: str, log_path: str) -> None:
        log_ip = node_ip.replace('.', '_')
        logging.basicConfig(filename=log_path + f'/local_controller_{log_ip}.log')
        logging.getLogger().setLevel(logging.INFO)
        self.node_ip = node_ip
        self.config = local_analyzer.load_config(config_path)
        self.record_buffer = local_analyzer.NcclRecord(self.config)
        self.global_controller_client = redis.StrictRedis(host=redis_ip, port=redis_port, db=0)
        self.reported_failslow_points = set()
    
    def report_failslow(self, failslow_rank, failslow_df):
        for i in range(len(failslow_df['ids'])):
            failslow_iter_id = str(failslow_rank) + "_" + str(failslow_df['ids'][i])
            if failslow_iter_id in self.reported_failslow_points:
                continue
            self.reported_failslow_points.add(failslow_iter_id)
            self.global_controller_client.rpush("failslow_ranks", str(failslow_rank))

    def report_profiling_result(self, profiling_result):
        for (comm_addr, comm_array) in profiling_result.items():
            comm_addr = str(comm_addr)
            duration_min = np.min(comm_array[:, 1])
            duration_max = np.max(comm_array[:, 1])
            duration_avg = np.mean(comm_array[:, 1])
            duration_std = np.std(comm_array[:, 1])
            result_str = f"{duration_min}_{duration_max}_{duration_avg}_{duration_std}"
            self.global_controller_client.set("Perf_" + comm_addr, result_str)

    def run(self):
        logging.critical(f"[Local controller] IP={self.node_ip} is launched!")
        while True:
            time.sleep(5)
            try:
                failslow_events = local_analyzer.detect_failslow(self.record_buffer, plot=False)
                if failslow_events is None:
                    logging.warning("No enough data is collected, please wait...")
                    continue
                failed_slow = False
                for global_rank, failslow_df in failslow_events.items():
                    if len(failslow_df) != 0:
                        failed_slow = True
                        logging.critical(
                            f"Failslow happens at rank={global_rank}, detail={failslow_df}")
                        self.report_failslow(global_rank, failslow_df)

                curr_state = self.global_controller_client.get("control_state").decode()
                # If other local controller reports fail-slow, we should also collect our profile results
                is_profiling = (curr_state == '1')
                # If the training is paused by global, we just wait until it is resumed
                is_validation = (curr_state == '2')

                # If a fail-slow is detected, we first report it to the global controller.
                # Then the global controller will notify each rank to start profiling, the
                # local side should wait it to collect some profile data and report it to global.
                # Finally, we should wait until the global controller finishes its validation.
                if failed_slow or is_profiling:
                    time.sleep(PROFILE_PERIOD)
                    profiling_res = local_analyzer.get_profile_results(self.record_buffer)
                    self.report_profiling_result(profiling_res)
                    # After the global controller's validation, it will reset the failslow_ranks
                    # This should notify each local controller to proceed.
                    while self.global_controller_client.llen("failslow_ranks") != 0:
                        logging.info("[Local Controller] Waiting for global validation...")
                        time.sleep(1)
                    # Now, the global validation is done, so we need to continue monitoring
                    logging.info("[Local Controller] Validation is done, resumed to monitoring mode!")
                    # We need to wait for the performance change to a normal value, and then clear the
                    # buffer, this is because if we don't wait, the performance gap between monitoring
                    # mode and profile mode would be a "performance gap", and will be recognized as a
                    # "false positive" fail-slow event
                    time.sleep(10)
                    self.record_buffer.clear()
                    self.reported_failslow_points = set()
                elif is_validation:
                    # wait until validation done
                    while curr_state != '0':
                        curr_state = self.global_controller_client.get("control_state").decode()
                        logging.info("[Local Controller] Wait because of pause")
                        time.sleep(1)
                    time.sleep(10)
                    self.record_buffer.clear()
                    self.reported_failslow_points = set()
            except Exception as e:
                logging.warning("Cannot detect failslow currently, there may be an error if it persists"\
                    + ", reason: " + str(e))


def start_local_controller():
    parser = argparse.ArgumentParser("Local side controller of fail-slow detection")
    parser.add_argument("-m", "--master_addr", default="127.0.0.1", type=str)
    parser.add_argument("-p", "--master_port", default=6379, type=int)
    parser.add_argument("-l", "--local_ip", default="127.0.0.1", type=str)
    parser.add_argument("-c", "--config_path", default="/workspace/ncclprobe/control_plane/config.json", type=str)
    parser.add_argument("-o", "--output_path", default="/workspace/ncclprobe/logs/", type=str)
    args = parser.parse_args()
    controller = LocalController(
        args.master_addr, args.master_port, args.local_ip, args.config_path, args.output_path)
    controller.run()


if __name__ == '__main__':
    start_local_controller()
