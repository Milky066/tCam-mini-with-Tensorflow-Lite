#ifndef TFLITE_FLOWER_MODEL_SETTINGS
#define TFLITE_FLOWER_MODEL_SETTINGS

#define imageSize 32
#define imageChannel 3

#define imageSizeInByte ((imageSize) * (imageSize) * (imageChannel))

#define labelCount 2

extern const char *labels[labelCount];

#endif