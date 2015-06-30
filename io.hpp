#ifndef __IO_HPP__860E2C25_1D2D_4C9E_8404_586174B4E717
#define __IO_HPP__860E2C25_1D2D_4C9E_8404_586174B4E717

#include <vector>
#include <string>
#include "common.h"


std::vector<std::string> generate_file_list(const config_t &config);
void saveaerdat(std::string filename, std::vector<dvs_event_t> &events);
void saveaerplain(std::string filename, std::vector<dvs_event_t> &events);
void saveedvstools(std::string filename, std::vector<dvs_event_t> &events);


#endif /* __IO_HPP__860E2C25_1D2D_4C9E_8404_586174B4E717 */

