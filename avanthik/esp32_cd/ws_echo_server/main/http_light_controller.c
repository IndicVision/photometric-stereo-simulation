/* PhotoStereo ESP32 — HTTP Light Controller (Station Mode)
 *
 * Architecture:
 *   Jetson is the Wi-Fi Access Point  (SSID: Photostereo_Jetson, 192.168.10.1)
 *   ESP32  is a Wi-Fi Station          (static IP: 192.168.10.2)
 *
 * Jetson controls LEDs via plain HTTP GET:
 *   GET http://192.168.10.2/light_on?id=N   → LED N ON, all others OFF  (N = 1..4)
 *   GET http://192.168.10.2/all_off          → All LEDs OFF
 *
 * LED Pin Map:
 *   LED 1 → GPIO 21
 *   LED 2 → GPIO 19
 *   LED 3 → GPIO 18
 *   LED 4 → GPIO 22
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_netif.h"
#include "esp_http_server.h"
#include "nvs_flash.h"
#include "driver/gpio.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "lwip/ip4_addr.h"

/* ─── Config ──────────────────────────────────────────────────────────────── */

#define WIFI_SSID "Photostereo_Jetson"
#define WIFI_PASSWORD "12345678"
/* No retry cap — ESP32 waits patiently until Jetson hotspot comes up */

/* Static IP for the ESP32 on the Jetson's hotspot */
#define STATIC_IP "192.168.10.2"
#define STATIC_GATEWAY "192.168.10.1"
#define STATIC_NETMASK "255.255.255.0"

#define LED_1 GPIO_NUM_21
#define LED_2 GPIO_NUM_19
#define LED_3 GPIO_NUM_18
#define LED_4 GPIO_NUM_22
#define NUM_LEDS 4

static const char *TAG = "photostereo";

/* FreeRTOS event group bit — set when IP is obtained */
#define WIFI_CONNECTED_BIT BIT0
static EventGroupHandle_t s_wifi_event_group;

/* ─── LED Helpers ─────────────────────────────────────────────────────────── */

static const gpio_num_t LED_PINS[NUM_LEDS] = {LED_1, LED_2, LED_3, LED_4};

static void init_led_gpios(void)
{
    for (int i = 0; i < NUM_LEDS; i++)
    {
        gpio_reset_pin(LED_PINS[i]);
        gpio_set_direction(LED_PINS[i], GPIO_MODE_OUTPUT);
        gpio_set_level(LED_PINS[i], 0);
    }
    ESP_LOGI(TAG, "All LED pins initialised LOW");
}

static void all_leds_off(void)
{
    for (int i = 0; i < NUM_LEDS; i++)
        gpio_set_level(LED_PINS[i], 0);
}

/* Turn on one LED (1-indexed), all others off */
static esp_err_t set_led(int id)
{
    if (id < 1 || id > NUM_LEDS)
        return ESP_ERR_INVALID_ARG;
    all_leds_off();
    gpio_set_level(LED_PINS[id - 1], 1);
    ESP_LOGI(TAG, "LED %d ON", id);
    return ESP_OK;
}

/* ─── HTTP Handlers ───────────────────────────────────────────────────────── */

/* GET /light_on?id=N */
static esp_err_t light_on_handler(httpd_req_t *req)
{
    char query[32] = {0};
    char id_str[8] = {0};

    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK &&
        httpd_query_key_value(query, "id", id_str, sizeof(id_str)) == ESP_OK)
    {
        int id = atoi(id_str);
        if (set_led(id) == ESP_OK)
        {
            httpd_resp_sendstr(req, "OK");
            return ESP_OK;
        }
    }

    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing or invalid id");
    return ESP_FAIL;
}

/* GET /all_off */
static esp_err_t all_off_handler(httpd_req_t *req)
{
    all_leds_off();
    ESP_LOGI(TAG, "All LEDs OFF");
    httpd_resp_sendstr(req, "OK");
    return ESP_OK;
}

/* GET /preview_on  — LED 1 on for framing/tuning before capture sequence */
static esp_err_t preview_on_handler(httpd_req_t *req)
{
    set_led(1);
    ESP_LOGI(TAG, "Preview: LED 1 ON");
    httpd_resp_sendstr(req, "OK");
    return ESP_OK;
}

/* GET /set_lights?ids=134 */
static esp_err_t set_lights_handler(httpd_req_t *req)
{
    char query[64] = {0};
    char ids_str[32] = {0};

    /* Ultra-safe ESP-IDF query extraction */
    size_t query_len = httpd_req_get_url_query_len(req) + 1;
    if (query_len > 1 && query_len <= sizeof(query))
    {
        if (httpd_req_get_url_query_str(req, query, query_len) == ESP_OK)
        {
            if (httpd_query_key_value(query, "ids", ids_str, sizeof(ids_str)) == ESP_OK)
            {

                /* Start with a clean slate */
                all_leds_off();

                /* Look for the digits in the squished string (e.g., "134") */
                if (strchr(ids_str, '1'))
                    gpio_set_level(LED_PINS[0], 1);
                if (strchr(ids_str, '2'))
                    gpio_set_level(LED_PINS[1], 1);
                if (strchr(ids_str, '3'))
                    gpio_set_level(LED_PINS[2], 1);
                if (strchr(ids_str, '4'))
                    gpio_set_level(LED_PINS[3], 1);

                ESP_LOGI(TAG, "Custom Preview Lights ON: %s", ids_str);
                httpd_resp_sendstr(req, "OK");
                return ESP_OK;
            }
        }
    }

    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing or invalid ids parameter");
    return ESP_FAIL;
}

static const httpd_uri_t uri_light_on = {
    .uri = "/light_on",
    .method = HTTP_GET,
    .handler = light_on_handler,
};

static const httpd_uri_t uri_all_off = {
    .uri = "/all_off",
    .method = HTTP_GET,
    .handler = all_off_handler,
};

static const httpd_uri_t uri_preview_on = {
    .uri = "/preview_on",
    .method = HTTP_GET,
    .handler = preview_on_handler,
};

static const httpd_uri_t uri_set_lights = {
    .uri = "/set_lights",
    .method = HTTP_GET,
    .handler = set_lights_handler,
};

/* ─── HTTP Server ─────────────────────────────────────────────────────────── */

static void start_http_server(void)
{
    httpd_handle_t server = NULL;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;

    if (httpd_start(&server, &config) == ESP_OK)
    {
        httpd_register_uri_handler(server, &uri_preview_on);
        httpd_register_uri_handler(server, &uri_light_on);
        httpd_register_uri_handler(server, &uri_all_off);

        /* Add our new endpoint here: */
        httpd_register_uri_handler(server, &uri_set_lights);

        ESP_LOGI(TAG, "HTTP server ready:");
        ESP_LOGI(TAG, "  http://%s/preview_on          -> LED 1 ON (setup/framing)", STATIC_IP);
        ESP_LOGI(TAG, "  http://%s/light_on?id=N       -> LED N ON, others OFF", STATIC_IP);
        ESP_LOGI(TAG, "  http://%s/set_lights?ids=1,3  -> Multi-LED ON, others OFF", STATIC_IP);
        ESP_LOGI(TAG, "  http://%s/all_off             -> All LEDs OFF", STATIC_IP);
    }
    else
    {
        ESP_LOGE(TAG, "Failed to start HTTP server");
    }
}

/* ─── Wi-Fi Event Handler ─────────────────────────────────────────────────── */

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START)
    {
        esp_wifi_connect();
    }
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED)
    {
        esp_wifi_connect(); /* retry immediately, forever */
        ESP_LOGW(TAG, "Disconnected — retrying...");
    }
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
    {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "------------------------------------");
        ESP_LOGI(TAG, "Connected to : %s", WIFI_SSID);
        ESP_LOGI(TAG, "ESP32 IP     : " IPSTR, IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "------------------------------------");
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

/* ─── Wi-Fi Station Init ──────────────────────────────────────────────────── */

static void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    /* Create default STA netif */
    esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();

    /* Stop DHCP client and assign static IP */
    ESP_ERROR_CHECK(esp_netif_dhcpc_stop(sta_netif));

    esp_netif_ip_info_t ip_info;
    memset(&ip_info, 0, sizeof(ip_info));
    ip4addr_aton(STATIC_IP, (ip4_addr_t *)&ip_info.ip);
    ip4addr_aton(STATIC_GATEWAY, (ip4_addr_t *)&ip_info.gw);
    ip4addr_aton(STATIC_NETMASK, (ip4_addr_t *)&ip_info.netmask);
    ESP_ERROR_CHECK(esp_netif_set_ip_info(sta_netif, &ip_info));

    /* Init Wi-Fi driver */
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASSWORD,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to \"%s\" ...", WIFI_SSID);

    /* Block until connected or hard failure */
    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_event_group,
        WIFI_CONNECTED_BIT,
        pdFALSE, pdFALSE,
        portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT)
    {
        ESP_LOGI(TAG, "Static IP active: %s", STATIC_IP);
    }

    /* Unregister one-shot handlers */
    ESP_ERROR_CHECK(esp_event_handler_instance_unregister(
        IP_EVENT, IP_EVENT_STA_GOT_IP, instance_got_ip));
    ESP_ERROR_CHECK(esp_event_handler_instance_unregister(
        WIFI_EVENT, ESP_EVENT_ANY_ID, instance_any_id));
    vEventGroupDelete(s_wifi_event_group);
}

/* ─── Main ────────────────────────────────────────────────────────────────── */

void app_main(void)
{
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    init_led_gpios();
    wifi_init_sta();
    start_http_server();

    /* FreeRTOS scheduler handles everything from here */
}