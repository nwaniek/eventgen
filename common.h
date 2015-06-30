#ifndef __COMMON_H__202D5AF2_41B1_4B7D_B6FA_F1094FEBBBB9
#define __COMMON_H__202D5AF2_41B1_4B7D_B6FA_F1094FEBBBB9


typedef enum {
	AEDAT = 0,
	PLAIN = 1,
	EDVSTOOLS = 2
} output_format_t;


/*
 * struct config_t - options passed to the application.
*/
typedef struct {
	char *file_pattern;
	char *file_target;
	output_format_t oformat;
	uint64_t frame_start;
	uint64_t frame_stop;
	uint64_t start_t;
	uint64_t delta_t;
	int thresh;
	bool warn_only;
} config_t;


/*
 * struct dvs_event_t - event representation. huge integers required due to CUDA
 */
typedef struct {
	uint8_t polarity;
	uint16_t x, y;
	uint64_t t;
} dvs_event_t;


#endif /* __COMMON_H__202D5AF2_41B1_4B7D_B6FA_F1094FEBBBB9 */

