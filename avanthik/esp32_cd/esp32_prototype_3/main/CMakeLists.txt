idf_component_register(SRCS "http_light_controller.c"
                    PRIV_REQUIRES esp_http_server esp_wifi nvs_flash esp_eth mdns
                    REQUIRES driver esp_driver_gpio
                    INCLUDE_DIRS ".")
