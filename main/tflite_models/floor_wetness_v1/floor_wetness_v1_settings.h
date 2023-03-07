#ifndef _FLOOR_WETNESS_V1_SETTINGS_H
#define _FLOOR_WETNESS_V1_SETTINGS_H

#define IMAGE_HEIGHT 30
#define IMAGE_WIDTH 40
#define IMAGE_CHANNEL 1

#define IMAGEE_SIZE_BYTE ((imageHeight) * (imageWidth) * (imageChannel))
#define MODEL_VERSION 1
#define LABEL_COUNT 4

extern const char *floor_label_list[LABEL_COUNT];

#endif