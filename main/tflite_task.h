#ifndef _TFLITE_TASK_H
#define _TFLITE_TASK_H

#ifdef __cplusplus
extern "C"
{
#endif
    void predict_image_from_buffer(int imageNumber);
    bool tflite_init();
    void tflite_task();
    void tflite_predict();
#ifdef __cplusplus
}
#endif

#endif