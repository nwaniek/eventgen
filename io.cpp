#include "io.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstring>
#include <cstdio>


/*
 * copied from edvstools to be able to save to the edvstools binary format
 */
namespace edvstools {
	typedef struct {
		uint64_t t;
		uint16_t x, y;
		uint8_t parity;
		uint8_t id;
	} edvs_event_t;
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


void
saveedvstools(std::string filename, std::vector<dvs_event_t> &events)
{
	using namespace std;

	// convert
	vector<edvstools::edvs_event_t> evs;
	for (auto &e: events) {
		evs.push_back({e.t, e.x, e.y, e.polarity, 0u});
	}

	// write
	FILE *fh = fopen(filename.c_str(), "w");
	size_t m = fwrite((const void*)evs.data(), sizeof(edvstools::edvs_event_t), evs.size(), fh);
	if (m != evs.size()) {
		std::cerr << "EE: could not write to file" << std::endl;
	}
	fclose(fh);
}
