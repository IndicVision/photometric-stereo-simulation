// import necessary libraries and define GPIO pins
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_log.h"

#define DATA_PIN GPIO_NUM_23
#define CLOCK_PIN GPIO_NUM_18
#define LATCH_PIN GPIO_NUM_5

static const char *TAG = "SR_CONTROL";

// Function to send 8 bits of data to the shift register
void send_byte(uint8_t data)
{
    for (int i = 0; i < 8; i++)
    {
        gpio_set_level(DATA_PIN, (data >> i) & 0x01);
        gpio_set_level(CLOCK_PIN, 1);
        esp_rom_delay_us(10);
        gpio_set_level(CLOCK_PIN, 0);
        esp_rom_delay_us(10);
    }
}

// Function to latch data from shift registers to output pins
void latch(void)
{
    gpio_set_level(LATCH_PIN, 1);
    esp_rom_delay_us(10);
    gpio_set_level(LATCH_PIN, 0);
}

// Helper to clear all LEDs immediately
void clear_all_leds(void)
{
    send_byte(0x00);
    send_byte(0x00);
    send_byte(0x00);
    latch();
}

// Logic to determine which shift register gets the data
void update_leds_ordered(uint8_t target_pattern, int group)
{
    if (group == 1) // SR3 (LEDs 1-8)
    {
        send_byte(target_pattern); // Send data first
        send_byte(0x00);           // Push it
        send_byte(0x00);           // Push it further
    }
    else if (group == 2) // SR2 (LEDs 9-16)
    {
        send_byte(0x00);
        send_byte(target_pattern); // Data lands in middle
        send_byte(0x00);
    }
    else // SR1 (LEDs 17-24)
    {
        send_byte(0x00);
        send_byte(0x00);
        send_byte(target_pattern); // Data stays at start
    }
    latch();
}

void init_gpio(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << DATA_PIN) | (1ULL << CLOCK_PIN) | (1ULL << LATCH_PIN),
        .mode = GPIO_MODE_OUTPUT,
    };
    gpio_config(&io_conf);
}

void app_main(void)
{
    init_gpio();

    // Initial clear
    clear_all_leds();

    // UART configuration for input
    uart_config_t uart_config = {
        .baud_rate = 115200,                             // Set baud rate in bps
        .data_bits = UART_DATA_8_BITS,                   // defines number of data bits for UART frame
        .parity = UART_PARITY_DISABLE,                   // defines the parity mode
        .stop_bits = UART_STOP_BITS_1,                   // defines the number of stop bits
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE};          // defines the flow control mode
    uart_driver_install(UART_NUM_0, 256, 0, 0, NULL, 0); // Install UART driver
    uart_param_config(UART_NUM_0, &uart_config);         // Configure UART parameters

    char input_buf[10]; // Buffer to hold user input
    int idx = 0;        // Index for input buffer

    ESP_LOGI(TAG, "System Ready.");
    ESP_LOGI(TAG, "Controls: Enter '1-24' for LED, '0' to Clear, 'q/Q' to Quit.");

    while (1)
    {
        uint8_t c;                                                     // Variable to hold received character
        if (uart_read_bytes(UART_NUM_0, &c, 1, pdMS_TO_TICKS(20)) > 0) // Read a byte from UART
        {
            // === PROCESS COMMAND (Enter Key) ===
            if (c == '\n' || c == '\r')
            {
                input_buf[idx] = '\0'; // Null-terminate the string
                putchar('\n');         // Echo newline for user feedback

                if (idx > 0) // Ensure there's input to process
                {
                    // === OPTION 1: STOP/KILL (Input 'q') ===
                    if (input_buf[0] == 'q' || input_buf[0] == 'Q')
                    {
                        ESP_LOGW(TAG, "Shutdown command received. Turning off LEDs and stopping task.");
                        clear_all_leds(); // Ensure lights are off before dying
                        break;            // Break the infinite while loop
                    }

                    int led_num = atoi(input_buf); // Convert input string to integer

                    // === OPTION 2: CLEAR (Input '0') ===
                    if (led_num == 0)
                    {
                        ESP_LOGI(TAG, "Command: CLEAR all LEDs");
                        clear_all_leds();
                    }
                    // === OPTION 3: TURN ON LED (Input 1-24) ===
                    else if (led_num >= 1 && led_num <= 24)
                    {
                        uint8_t pattern = 0; // Variable to hold LED pattern
                        int group = 0;       // Variable to hold group number

                        // Identify the group and pattern exactly like original code
                        if (led_num <= 8)
                        {
                            group = 1;
                            pattern = (0x80 >> (led_num - 1));
                        }
                        else if (led_num <= 16)
                        {
                            group = 2;
                            pattern = (0x80 >> (led_num - 9));
                        }
                        else
                        {
                            group = 3;
                            pattern = (0x80 >> (led_num - 17));
                        }

                        ESP_LOGI(TAG, "LED %d ON (Pattern 0x%02X in Group %d)", led_num, pattern, group);
                        update_leds_ordered(pattern, group);
                    }
                    else
                    {
                        ESP_LOGE(TAG, "Invalid: %d", led_num);
                    }
                }
                idx = 0; // Reset index for next input
            }
            // === COLLECT INPUT ===
            // Allow digits (0-9) OR letters 'q'/'Q' into the buffer
            else if (((c >= '0' && c <= '9') || c == 'q' || c == 'Q') && idx < 9)
            {
                input_buf[idx++] = c; // Store character and increment index
                putchar(c);           // Echo back the character to terminal for user to see
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10)); // Small delay to prevent CPU overload
    }

    // Code reaches here only if 'q' was pressed
    ESP_LOGE(TAG, "System Halted. Reboot required to restart.");
    vTaskDelete(NULL); // Delete this task completely
}