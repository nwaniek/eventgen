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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


typedef struct {
	char *file_pattern;
	uint64_t frame_start;
	uint64_t frame_stop;
	uint64_t start_t;
	uint64_t delta_t;
	bool warn_only;
} config_t;


static const char *usage =
"Usage: eventgen [options] file-pattern start stop\n"
"file pattern may be something like '/path/to/files/%05d.png'\n"
"and will be replaced according to the start and stop numbers.\n"
"Options: \n"
"  -t T   set the start time\n"
"  -d D   set the time delta between consecutive frames\n"
"  -h     show this help\n"
"  -w     only emit warning if a file is missing. default: exit\n"
"The output is stored as event data in evdat format.";


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
	return 0;
}


/*
 * test if a file exists. return 0 if the file exists and 1 if not. if warn_only
 * is set in the config, the function will always return 0.
 */
int
test_file(const config_t &config, char *filename)
{
	struct stat sb;
	if (stat(filename, &sb) == -1) {
		if (errno == ENOENT) {
			if (config.warn_only)
				std::cerr << "WW: File ";
			else
				std::cerr << "EE: File ";
			std::cerr << filename << "' does not exist." << std::endl;
			return config.warn_only ? 0 : 1;
		}
		else {
			if (config.warn_only)
				std::cerr << "WW: File ";
			else
				std::cerr << "EE: File ";
			std::cerr << filename << "' does not exist." << std::endl;
			return config.warn_only ? 0 : 1;
		}
	}
	return 0;
}


int
process_files(std::vector<std::string> &files)
{
	// pass all the files to the CUDA kernel

	/*
	const char *fname = "/home/rochus/data/FlorianScherer-OpticFlow-ImageSequences/Frames/0000.png";

	int x, y, n;
	unsigned char *data = stbi_load(fname, &x, &y, &n, 0);
	std::cout << x << ", " << y << ", " << n << std::endl;

	stbi_image_free(data);

	// call_wrapper();
	return EXIT_SUCCESS;
	*/

	return EXIT_SUCCESS;
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
		if (test_file(config, buffer)) exit(EXIT_FAILURE);
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
	return process_files(files);
}