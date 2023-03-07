/*
 * tCam-Mini Main
 *
 *
 * Copyright 2020-2022 Dan Julio
 *
 * This file is part of tCam.
 *
 * tCam is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * tCam is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with tCam.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "sys_utilities.h"
#include <stdio.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "net_cmd_task.h"
#include "sif_cmd_task.h"
#include "ctrl_task.h"
#include "lep_task.h"
#include "mon_task.h"
#include "rsp_task.h"
#include "system_config.h"

#include "tflite_task.h"

static const char *TAG = "main";
StaticTask_t rsp_task_tcb;
StackType_t *rsp_task_stack;

void app_main(void)
{
    int brd_type;
    int if_mode;

    ESP_LOGI(TAG, "tCamMini starting");

    // Start the control task to light the red light immediately
    // and to determine what kind of interface we will be using
    xTaskCreatePinnedToCore(&ctrl_task, "ctrl_task", 2176, NULL, 1, &task_handle_ctrl, 0);
    // Allow task to start and determine operating mode
    vTaskDelay(pdMS_TO_TICKS(50));
    ctrl_get_if_mode(&brd_type, &if_mode); // Set the operation mode for brd_type and if_mode
    ESP_LOGI(TAG, "brd_type: %d, if_mode: %d", brd_type, if_mode);
    // Initialize the SPI and I2C drivers
    if (!system_esp_io_init(brd_type, if_mode))
    {
        ESP_LOGE(TAG, "ESP32 init failed");
        ctrl_set_fault_type(CTRL_FAULT_ESP32_INIT);
        while (1)
        {
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }

    // Initialize the camera's peripheral devices
    if (!system_peripheral_init(brd_type, if_mode))
    {
        ESP_LOGE(TAG, "Peripheral init failed");
        ctrl_set_fault_type(CTRL_FAULT_PERIPH_INIT);
        while (1)
        {
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }

    // Pre-allocate big buffers

    if (!system_buffer_init())
    {
        ESP_LOGE(TAG, "Memory allocate failed");
        ctrl_set_fault_type(CTRL_FAULT_MEM_INIT);
        while (1)
        {
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }

    // Delay for Lepton internal initialization on power-on (max 950 mSec)
    vTaskDelay(pdMS_TO_TICKS(900));

    // Notify control task that we've successfully started up
    xTaskNotify(task_handle_ctrl, CTRL_NOTIFY_STARTUP_DONE, eSetBits);

    if (!tflite_init())
    {
        ESP_LOGE(TAG, "TfLite initilisation fails");
    }
    rsp_task_stack = heap_caps_calloc(1, sizeof(StackType_t) * 1024 * 8, MALLOC_CAP_SPIRAM);
    // Start tasks
    // 30/1/2023: Add ML task into Core 0,
    //  Core 0 : PRO - everything but lepton task
    //  Core 1 : APP - lepton task
    if (if_mode == CTRL_IF_MODE_SIF)
    {
        xTaskCreatePinnedToCore(sif_cmd_task, "sif_cmd_task", 3072, NULL, 1, &task_handle_cmd, 0);
        xTaskCreatePinnedToCore(rsp_task, "rsp_task", 2816, NULL, 19, &task_handle_rsp, 0);
        xTaskCreatePinnedToCore(lep_task, "lep_task", 2048, NULL, 18, &task_handle_lep, 1);
    }
    else // 30/1/2022: tCam rev2 uses this below
    {

        xTaskCreatePinnedToCore(net_cmd_task, "net_cmd_task", 3072, NULL, 1, &task_handle_cmd, 0); // 3072
        // Uncomment below to include tflite_task into the prediction loop
        xTaskCreatePinnedToCore(lep_task, "lep_task", 2048, NULL, 19, &task_handle_lep, 1);
        // xTaskCreatePinnedToCore(rsp_task, "rsp_task", 2816, NULL, 19, &task_handle_rsp, 0); // 2816 Stack depth
        task_handle_rsp = xTaskCreateStaticPinnedToCore(rsp_task, "rsp_task", 1024 * 8, NULL, 19, rsp_task_stack, &rsp_task_tcb, 0);
        // 21/2/2023: Currently the task is idle
        // xTaskCreatePinnedToCore(&tflite_task, "tflift_task", 2048, NULL, 18, &task_handle_tflite, 0);
    }

#ifdef INCLUDE_SYS_MON
    xTaskCreatePinnedToCore(&mon_task, "mon_task", 2048, NULL, 1, &task_handle_mon, 0);
#endif
}
