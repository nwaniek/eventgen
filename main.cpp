#include <cuda.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <memory>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include "dvs.h"
#include "common.h"


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
"  -f format     output format. one of 'aedat', 'plain'. default: aedat\n"
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
	while ((opt = getopt(argc, argv, "cwtT::d:hf:")) != -1) {
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


inline void
write_bigendian(std::ofstream &f, uint32_t i)
{
	uint8_t buf[4];
	buf[0] = (i & 0xff000000) >> 24;
	buf[1] = (i & 0x00ff0000) >> 16;
	buf[2] = (i & 0x0000ff00) >> 8;
	buf[3] = (i & 0x000000ff);
	f.write((char*)&buf, sizeof(buf));
}


void
saveaerdat(std::string filename, std::vector<dvs_event_t> &events)
{
	using namespace std;

	ofstream f;
	f.open(filename, std::ios_base::out | std::ios_base::binary);

	const char header[] =
		"#!AER-DAT2.0\r\n"
		"# This is a raw AE data file created by saveaerdat.m\r\n"
		"# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n"
		"# Timestamps tick is 1 us\r\n";

	f.write(header, strlen(header));
	for (auto &e: events) {
		// TODO: check if the computation is correct
		uint32_t addr = (2 << 21) * e.y + (240-1-e.x) * (2 << 11) + (1-e.polarity) * (2 << 10);
		uint32_t t = (uint32_t)e.t;
		write_bigendian(f, addr);
		write_bigendian(f, t);
	}
	f.close();
}


void
saveaerplain(std::string filename, std::vector<dvs_event_t> &events)
{
	using namespace std;

	ofstream f;
	f.open(filename, std::ios_base::out);

	for (auto &e: events)
		f << (int)e.t << " " << (int)e.polarity << " " << (int)e.x << " " << (int)e.y << std::endl;
	f.close();
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

	// generate list of files (and check if all files are available) and run
	// the CUDA kernel on pairs of images.
	auto files = generate_file_list(config);
	vector<dvs_event_t> events = process_files(config, files);

	switch (config.oformat) {
	case AEDAT:
		saveaerdat(config.file_target, events);
		break;
	case PLAIN:
		saveaerplain(config.file_target, events);
		break;
	}
	return EXIT_SUCCESS;
}
