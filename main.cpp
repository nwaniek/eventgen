#include <cuda.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include "kernels/dvs.h"
#include "kernels/misc.h"
#include "common.h"
#include "io.hpp"


static const char *usage =
"Usage: eventgen [options] file-pattern start stop file-name\n"
"Positional Arguments:\n"
"  file-pattern  pattern to search files for, e.g. '/path/%05d.png'\n"
"  start         start index to be used in file-pattern.\n"
"  stop          stop index to be used in file-pattern.\n"
"  file-name     filename to write result to\n"
"Options: \n"
"  -t time       set the start time\n"
"  -T threshold  set the pixel-threshod. default = 10\n"
"  -d D          set the time delta between consecutive frames\n"
"  -h            show this help\n"
"  -w            only emit warning if a file is missing. default: exit\n"
"  -f format     output format. one of 'aedat', 'plain', 'edvstools'. default: aedat\n"
"The output is stored as event data in evdat format.";


/*
 * parse_args - parse arguments into config structure.
 */
int
parse_args(config_t &config, int argc, char *argv[])
{
	config.start_t = 0;
	config.delta_t = 1;
	config.warn_only = false;
	config.thresh = 10;
	config.oformat = AEDAT;

	int opt;
	while ((opt = getopt(argc, argv, "cwt:T:d:hf:")) != -1) {
		switch (opt) {
		case 'c':
			print_cuda_info();
			exit(EXIT_SUCCESS);
			break;
		case 't':
			config.start_t = boost::lexical_cast<uint64_t>(optarg);
			break;
		case 'd':
			config.delta_t = boost::lexical_cast<uint64_t>(optarg);
			break;
		case 'w':
			config.warn_only = true;
			break;
		case 'T':
			config.thresh = boost::lexical_cast<int>(optarg);
			break;
		case 'f':
			if (!strcmp(optarg, "plain"))
				config.oformat = PLAIN;
			else if (!strcmp(optarg, "aedat"))
				config.oformat = AEDAT;
			else if (!strcmp(optarg, "edvstools"))
				config.oformat = EDVSTOOLS;
			else {
				std::cerr << "EE: unknown output format '" << optarg << "'." << std::endl;
				return 1;
			}
			break;
		case 'h':
			return 1;
		default:
			return 1;
		}
	}

	// check if we got enough arguments
	if ((argc - optind) < 4) {
		std::cerr << "EE: insufficient arguments" << std::endl;
		return 1;
	}

	config.file_pattern = argv[optind];
	try {
		config.frame_start  = boost::lexical_cast<uint64_t>(argv[optind + 1]);
		config.frame_stop   = boost::lexical_cast<uint64_t>(argv[optind + 2]);
	}
	catch (boost::bad_lexical_cast&) {
		std::cerr << "EE: start/stop need to be in integer format" << std::endl;
		return 1;
	}
	config.file_target = argv[optind + 3];

	if (config.frame_start >= config.frame_stop) {
		std::cerr << "EE: start index > stop index" << std::endl;
		return 1;
	}

	return 0;
}


int
main(int argc, char *argv[])
{
	using namespace std;
	initCuda();

	// read in the configuration
	config_t config;
	if (parse_args(config, argc, argv)) {
		std::cout << usage << std::endl;
		return EXIT_FAILURE;
	}

	// generate list of files (and check if all files are available) and run
	// the CUDA kernel on pairs of images.
	auto files = generate_file_list(config);
	vector<dvs_event_t> events = process_files(config, files);
	std::sort(events.begin(), events.end(), [](dvs_event_t &a, dvs_event_t &b) {
				return a.t < b.t;
			});

	switch (config.oformat) {
	case AEDAT:
		saveaerdat(config.file_target, events);
		break;
	case PLAIN:
		saveaerplain(config.file_target, events);
		break;
	case EDVSTOOLS:
		saveedvstools(config.file_target, events);
		break;
	}
	return EXIT_SUCCESS;
}
