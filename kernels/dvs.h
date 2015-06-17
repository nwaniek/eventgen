#ifndef __DVS_H__B0AC2419_E92A_41B5_9536_F492EE70067A
#define __DVS_H__B0AC2419_E92A_41B5_9536_F492EE70067A

#include <vector>
#include <string>
#include "common.h"

std::vector<dvs_event_t> process_files(config_t &config, std::vector<std::string> &files);
void print_cuda_info();

#endif /* __DVS_H__B0AC2419_E92A_41B5_9536_F492EE70067A */

