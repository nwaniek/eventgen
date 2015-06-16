#include <cuda.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <memory>
#include <boost/lexical_cast.hpp>
#include "dvs.h"
#include "common.h"


static const char *usage =
"Usage: eventgen [options] file-pattern start stop\n"
"Positional Arguments:\n"
"  file-pattern  pattern to search files for, e.g. '/path/%05d.png'\n"
"  start         start index to be used in file-pattern.\n"
"  stop          stop index to be used in file-pattern.\n"
"Options: \n"
"  -t T          set the start time\n"
"  -d D          set the time delta between consecutive frames\n"
"  -h            show this help\n"
"  -w            only emit warning if a file is missing. default: exit\n"
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

	int opt;
	while ((opt = getopt(argc, argv, "wt:d:h")) != -1) {
		switch (opt) {
		case 't':
			config.start_t = atoi(optarg);
			break;
		case 'd':
			config.delta_t = atoi(optarg);
			break;
		case 'w':
			config.warn_only = true;
			break;
		case 'h':
			return 1;
		default:
			return 1;
		}
	}

	// check if we got enough arguments
	if ((argc - optind) < 3) {
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

	if (config.frame_start >= config.frame_stop) {
		std::cerr << "EE: start index > stop index" << std::endl;
		return 1;
	}

	return 0;
}


/*
 * test_file - see if a file exists. returns 0 if the file exists and 1 if not.
 */
int
test_file(char *filename)
{
	struct stat sb;
	if (stat(filename, &sb) == -1)
		return 1;
	return 0;
}


std::vector<std::string>
generate_file_list(const config_t &config)
{
	using namespace std;

	// check if all files exist
	vector<string> files;
	for (uint64_t i = config.frame_start; i < config.frame_stop; i++) {
		char buffer[512] = {0};
		snprintf(buffer, 512, config.file_pattern, i);
		if (test_file(buffer)) {
			if (config.warn_only)
				std::cerr << "WW: File '" << buffer << "' does not exist or is unavailable." << std::endl;
			else {
				std::cerr << "EE: File '" << buffer << "' does not exist or is unavailable." << std::endl;
				exit(EXIT_FAILURE);
			}
		}
		files.push_back(string(buffer));
	}
	return std::move(files);
}


int
main(int argc, char *argv[])
{
	using namespace std;

	// read in the configuration
	config_t config;
	if (parse_args(config, argc, argv)) {
		std::cout << usage << std::endl;
		return EXIT_FAILURE;
	}

	auto files = generate_file_list(config);
	return process_files(config, files);
}
