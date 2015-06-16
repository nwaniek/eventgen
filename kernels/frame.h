#ifndef __FRAME_H__65F2E4AA_A035_45CC_8E39_827B1621E122
#define __FRAME_H__65F2E4AA_A035_45CC_8E39_827B1621E122

#include <string>

/**
 * struct frame - load a frame from file and load it to memory.
 *
 * the frame struct will load the frame from a file. If necessary, it will
 * convert the file to grayscale, and then keep it in device memory.
 */
struct Frame
{
	~Frame();
	int load_from_file(std::string filename);

	std::string filename;
	int x, y, n;
	size_t memsize;
	unsigned char *data = nullptr;
	unsigned char *dev_data = nullptr;
};


#endif /* __FRAME_H__65F2E4AA_A035_45CC_8E39_827B1621E122 */

