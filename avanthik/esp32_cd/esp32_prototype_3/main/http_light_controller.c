/* PhotoStereo ESP32 — HTTP Light Controller
 *
 * Architecture (Laptop-as-Server):
 *   User's phone creates a Wi-Fi hotspot (any name / any IP)
 *   Laptop joins that hotspot, runs the PhotoStereo server,
 *     and announces itself as  photostereo.local  via mDNS
 *   ESP32 joins the same hotspot (DHCP, dynamic IP) and
 *     announces itself as  esp32ps.local  so the laptop can find it
 *
 * Provisioning (first-time or re-provisioning):
 *   Option 1 — /reprovision endpoint (RECOMMENDED):
 *     While ESP32 is running normally, open a browser on your phone
 *     (connected to the same hotspot) and go to:
 *         http://esp32ps.local/reprovision
 *     The ESP32 erases its saved credentials and reboots into provisioning
 *     mode automatically. No physical button needed.
 *
 *   Option 2 — Auto-fallback:
 *     If the saved hotspot is unreachable and the ESP32 fails to connect
 *     MAX_RETRY_BEFORE_PROVISION times (~45 seconds), it automatically
 *     erases credentials and enters provisioning mode on its own.
 *
 *   Option 3 — BOOT button (hardware fallback only):
 *     Hold GPIO 0 (BOOT button) for >= 3s at power-on to force
 *     provisioning mode. Use this only if the above two options fail.
 *
 *   Provisioning procedure (same for all three options):
 *     Step 1: Connect phone to Wi-Fi "PhotoStereo-Setup" (open, no password)
 *     Step 2: Open phone browser -> http://192.168.4.1
 *     Step 3: Fill in hotspot SSID + password, tap Save
 *     Step 4: ESP32 reboots and joins your hotspot automatically
 *
 *   (Alternatively, POST JSON to http://192.168.4.1/provision
 *    Body: { "ssid": "YourHotspot", "password": "YourPass" })
 *
 * LED Pin Map:
 *   LED 1 -> GPIO 21
 *   LED 2 -> GPIO 19
 *   LED 3 -> GPIO 18
 *   LED 4 -> GPIO 22
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
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
#include "nvs.h"
#include "driver/gpio.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "mdns.h"

/* ─── Config ──────────────────────────────────────────────────────────────── */

#define PROV_AP_SSID        "PhotoStereo-Setup"
#define PROV_AP_IP          "192.168.4.1"

#define NVS_NAMESPACE       "ps_wifi"
#define NVS_KEY_SSID        "ssid"
#define NVS_KEY_PASSWORD    "password"

#define MDNS_HOSTNAME       "esp32ps"
#define MDNS_INSTANCE       "PhotoStereo ESP32"

#define BOOT_BUTTON_GPIO    GPIO_NUM_0
#define BOOT_HOLD_MS        3000

/* How many consecutive failed connection attempts before giving up and
 * self-reprovisioning. Each attempt takes ~3 s, so 15 = ~45 seconds. */
#define MAX_RETRY_BEFORE_PROVISION  15

#define LED_1 GPIO_NUM_21
#define LED_2 GPIO_NUM_19
#define LED_3 GPIO_NUM_18
#define LED_4 GPIO_NUM_22
#define NUM_LEDS 4

static const char *TAG = "photostereo";

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1
static EventGroupHandle_t s_wifi_event_group;

/* Retry counter — incremented on every DISCONNECTED event, reset on GOT_IP */
static int s_retry_count = 0;

/* ─── LED Helpers ─────────────────────────────────────────────────────────── */

static const gpio_num_t LED_PINS[NUM_LEDS] = {LED_1, LED_2, LED_3, LED_4};

static void init_led_gpios(void)
{
    for (int i = 0; i < NUM_LEDS; i++) {
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

static esp_err_t set_led(int id)
{
    if (id < 1 || id > NUM_LEDS) return ESP_ERR_INVALID_ARG;
    all_leds_off();
    gpio_set_level(LED_PINS[id - 1], 1);
    ESP_LOGI(TAG, "LED %d ON", id);
    return ESP_OK;
}

/* ─── NVS Credential Storage ──────────────────────────────────────────────── */

static esp_err_t nvs_read_creds(char *ssid, size_t ssid_len,
                                 char *password, size_t pass_len)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READONLY, &h);
    if (err != ESP_OK) return err;

    err = nvs_get_str(h, NVS_KEY_SSID, ssid, &ssid_len);
    if (err == ESP_OK)
        err = nvs_get_str(h, NVS_KEY_PASSWORD, password, &pass_len);

    nvs_close(h);
    return err;
}

static esp_err_t nvs_write_creds(const char *ssid, const char *password)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h);
    if (err != ESP_OK) return err;

    err = nvs_set_str(h, NVS_KEY_SSID, ssid);
    if (err == ESP_OK)
        err = nvs_set_str(h, NVS_KEY_PASSWORD, password);
    if (err == ESP_OK)
        err = nvs_commit(h);

    nvs_close(h);
    return err;
}

static void nvs_erase_creds(void)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h) == ESP_OK) {
        nvs_erase_key(h, NVS_KEY_SSID);
        nvs_erase_key(h, NVS_KEY_PASSWORD);
        nvs_commit(h);
        nvs_close(h);
        ESP_LOGI(TAG, "NVS credentials erased");
    }
}

/* ─── Boot-button check ───────────────────────────────────────────────────── */

static bool boot_button_held(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << BOOT_BUTTON_GPIO),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);

    if (gpio_get_level(BOOT_BUTTON_GPIO) != 0)
        return false;

    ESP_LOGI(TAG, "BOOT button held — waiting %d ms to confirm...", BOOT_HOLD_MS);
    vTaskDelay(pdMS_TO_TICKS(BOOT_HOLD_MS));
    return (gpio_get_level(BOOT_BUTTON_GPIO) == 0);
}

/* ─── URL Decode Helpers ──────────────────────────────────────────────────── */

static int hex_to_byte(char hi, char lo)
{
    int h = (hi >= '0' && hi <= '9') ? hi - '0' :
            (hi >= 'A' && hi <= 'F') ? hi - 'A' + 10 :
            (hi >= 'a' && hi <= 'f') ? hi - 'a' + 10 : -1;
    int l = (lo >= '0' && lo <= '9') ? lo - '0' :
            (lo >= 'A' && lo <= 'F') ? lo - 'A' + 10 :
            (lo >= 'a' && lo <= 'f') ? lo - 'a' + 10 : -1;
    if (h < 0 || l < 0) return -1;
    return (h << 4) | l;
}

static void url_decode(char *dst, const char *src, size_t dst_len)
{
    size_t i = 0, j = 0;
    while (src[i] && j < dst_len - 1) {
        if (src[i] == '%' && src[i+1] && src[i+2]) {
            int b = hex_to_byte(src[i+1], src[i+2]);
            if (b >= 0) {
                dst[j++] = (char)b;
                i += 3;
                continue;
            }
        }
        if (src[i] == '+') {
            dst[j++] = ' ';
            i++;
            continue;
        }
        dst[j++] = src[i++];
    }
    dst[j] = '\0';
}

static void form_get_value(const char *body, const char *key,
                            char *out, size_t out_len)
{
    char search[64];
    snprintf(search, sizeof(search), "%s=", key);
    const char *p = strstr(body, search);
    if (!p) { out[0] = '\0'; return; }
    p += strlen(search);
    const char *end = strchr(p, '&');
    size_t len = end ? (size_t)(end - p) : strlen(p);
    if (len >= out_len) len = out_len - 1;
    char encoded[256] = {0};
    strncpy(encoded, p, len);
    url_decode(out, encoded, out_len);
}

/* ─── HTTP Handlers ───────────────────────────────────────────────────────── */

/* GET /provision — serves the HTML setup form */
static esp_err_t provision_page_handler(httpd_req_t *req)
{
    const char *html =
        "<!DOCTYPE html><html><head>"
        "<title>PhotoStereo Setup</title>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<style>"
        "body{font-family:sans-serif;max-width:420px;margin:40px auto;padding:20px}"
        "h2{color:#2196F3;margin-bottom:4px}"
        "p{color:#666;font-size:14px;margin-top:4px}"
        "label{display:block;margin-top:16px;font-weight:bold;font-size:14px}"
        "input{width:100%;padding:12px;margin-top:6px;box-sizing:border-box;"
               "font-size:16px;border:1px solid #ccc;border-radius:6px}"
        "button{width:100%;padding:14px;margin-top:24px;background:#4CAF50;"
               "color:white;border:none;border-radius:6px;font-size:16px;"
               "cursor:pointer;font-weight:bold}"
        "button:active{background:#388E3C}"
        ".note{font-size:12px;color:#999;margin-top:16px;text-align:center}"
        "</style></head><body>"
        "<h2>&#128247; PhotoStereo Setup</h2>"
        "<p>Enter your <strong>laptop hotspot</strong> details below.</p>"
        "<form method='POST' action='/provision'>"
        "<label>Hotspot Name (SSID):</label>"
        "<input type='text' name='ssid' placeholder='My Laptop Hotspot' required>"
        "<label>Password:</label>"
        "<input type='password' name='password' placeholder='(leave blank if open)'>"
        "<button type='submit'>Save and Connect &#9654;</button>"
        "</form>"
        "<p class='note'>LED 1 blinks = setup mode. Stops blinking = connected.</p>"
        "</body></html>";

    httpd_resp_set_type(req, "text/html");
    httpd_resp_sendstr(req, html);
    return ESP_OK;
}

/* POST /provision — receives credentials from either HTML form or JSON */
static esp_err_t provision_handler(httpd_req_t *req)
{
    char buf[512] = {0};
    int received  = httpd_req_recv(req, buf, sizeof(buf) - 1);
    if (received <= 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Empty body");
        return ESP_FAIL;
    }
    buf[received] = '\0';
    ESP_LOGI(TAG, "Provision body: %.100s", buf);

    char ssid[64]     = {0};
    char password[64] = {0};

    if (buf[0] == '{') {
        /* ── JSON path ──────────────────────────────────────────────── */
        char *p = strstr(buf, "\"ssid\"");
        if (p) {
            p = strchr(p + 6, ':');
            if (p) { p = strchr(p, '"'); if (p) { p++;
                char *end = strchr(p, '"');
                if (end) { size_t l = (size_t)(end - p);
                    if (l >= sizeof(ssid)) l = sizeof(ssid) - 1;
                    strncpy(ssid, p, l); }}}
        }
        p = strstr(buf, "\"password\"");
        if (p) {
            p = strchr(p + 10, ':');
            if (p) { p = strchr(p, '"'); if (p) { p++;
                char *end = strchr(p, '"');
                if (end) { size_t l = (size_t)(end - p);
                    if (l >= sizeof(password)) l = sizeof(password) - 1;
                    strncpy(password, p, l); }}}
        }
    } else {
        /* ── URL-encoded form path (browser HTML form) ──────────────── */
        form_get_value(buf, "ssid",     ssid,     sizeof(ssid));
        form_get_value(buf, "password", password, sizeof(password));
    }

    if (strlen(ssid) == 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing ssid");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Provisioning with ssid=\"%s\"", ssid);
    esp_err_t err = nvs_write_creds(ssid, password);
    if (err != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "NVS write failed");
        return ESP_FAIL;
    }

    const char *accept = NULL;
    char accept_buf[64] = {0};
    if (httpd_req_get_hdr_value_str(req, "Accept",
                                    accept_buf, sizeof(accept_buf)) == ESP_OK) {
        accept = accept_buf;
    }
    bool wants_html = (accept == NULL || strstr(accept, "text/html") != NULL);

    if (wants_html) {
        httpd_resp_set_type(req, "text/html");
        httpd_resp_sendstr(req,
            "<!DOCTYPE html><html><body style='"
            "font-family:sans-serif;max-width:420px;margin:40px auto;padding:20px'>"
            "<h2 style='color:#4CAF50'>&#10003; Saved!</h2>"
            "<p>ESP32 is now connecting to your hotspot...</p>"
            "<p>LED 1 will stop blinking once connected.</p>"
            "<p style='color:#999;font-size:13px'>"
            "You can now disconnect from <strong>PhotoStereo-Setup</strong> "
            "and connect to your normal hotspot.</p>"
            "</body></html>");
    } else {
        httpd_resp_set_type(req, "application/json");
        httpd_resp_sendstr(req,
            "{\"status\":\"OK\",\"message\":\"Credentials saved. Rebooting...\"}");
    }

    vTaskDelay(pdMS_TO_TICKS(500));
    esp_restart();
    return ESP_OK;
}

/* GET /reprovision
 *
 * Call this from your phone browser while the ESP32 is running normally:
 *     http://esp32ps.local/reprovision
 *
 * The ESP32 will:
 *   1. Send a plain-text confirmation page back to the browser
 *   2. Wait 500 ms so the response can flush to the browser
 *   3. Erase the saved WiFi credentials from NVS flash
 *   4. Reboot into provisioning mode (LED 1 blinking, AP = PhotoStereo-Setup)
 *
 * After rebooting:
 *   - Connect your phone to WiFi "PhotoStereo-Setup"
 *   - Open http://192.168.4.1 in your browser
 *   - Enter new hotspot credentials and tap Save
 */
static esp_err_t reprovision_handler(httpd_req_t *req)
{
    ESP_LOGI(TAG, "/reprovision called — erasing credentials and rebooting.");

    httpd_resp_set_type(req, "text/html");
    httpd_resp_sendstr(req,
        "<!DOCTYPE html><html><body style='"
        "font-family:sans-serif;max-width:420px;margin:40px auto;padding:20px'>"
        "<h2 style='color:#FF9800'>&#8635; Reprovisioning...</h2>"
        "<p>Erasing saved WiFi credentials.</p>"
        "<p>The ESP32 will reboot in a moment.</p>"
        "<hr>"
        "<p><strong>Next steps:</strong></p>"
        "<ol>"
        "<li>Connect your phone to WiFi: <strong>PhotoStereo-Setup</strong></li>"
        "<li>Open: <strong>http://192.168.4.1</strong></li>"
        "<li>Enter new hotspot credentials and tap Save</li>"
        "</ol>"
        "<p style='color:#999;font-size:13px'>"
        "LED 1 will start blinking to confirm setup mode.</p>"
        "</body></html>");

    /* Give the HTTP response time to fully flush to the browser
     * before we pull the rug out with esp_restart(). */
    vTaskDelay(pdMS_TO_TICKS(500));

    nvs_erase_creds();
    esp_restart();
    return ESP_OK;
}

/* GET /light_on?id=N */
static esp_err_t light_on_handler(httpd_req_t *req)
{
    char query[32] = {0};
    char id_str[8] = {0};

    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK &&
        httpd_query_key_value(query, "id", id_str, sizeof(id_str)) == ESP_OK)
    {
        int id = atoi(id_str);
        if (set_led(id) == ESP_OK) {
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

/* GET /preview_on */
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
    char query[64]   = {0};
    char ids_str[32] = {0};

    size_t query_len = httpd_req_get_url_query_len(req) + 1;
    if (query_len > 1 && query_len <= sizeof(query)) {
        if (httpd_req_get_url_query_str(req, query, query_len) == ESP_OK) {
            if (httpd_query_key_value(query, "ids", ids_str, sizeof(ids_str)) == ESP_OK) {
                all_leds_off();
                if (strchr(ids_str, '1')) gpio_set_level(LED_PINS[0], 1);
                if (strchr(ids_str, '2')) gpio_set_level(LED_PINS[1], 1);
                if (strchr(ids_str, '3')) gpio_set_level(LED_PINS[2], 1);
                if (strchr(ids_str, '4')) gpio_set_level(LED_PINS[3], 1);
                ESP_LOGI(TAG, "Custom Preview Lights ON: %s", ids_str);
                httpd_resp_sendstr(req, "OK");
                return ESP_OK;
            }
        }
    }
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing or invalid ids");
    return ESP_FAIL;
}

/* ─── URI Registrations ───────────────────────────────────────────────────── */

static const httpd_uri_t uri_provision_page = {
    .uri="/provision",    .method=HTTP_GET,  .handler=provision_page_handler };
static const httpd_uri_t uri_provision      = {
    .uri="/provision",    .method=HTTP_POST, .handler=provision_handler      };
static const httpd_uri_t uri_light_on       = {
    .uri="/light_on",     .method=HTTP_GET,  .handler=light_on_handler       };
static const httpd_uri_t uri_all_off        = {
    .uri="/all_off",      .method=HTTP_GET,  .handler=all_off_handler        };
static const httpd_uri_t uri_preview_on     = {
    .uri="/preview_on",   .method=HTTP_GET,  .handler=preview_on_handler     };
static const httpd_uri_t uri_set_lights     = {
    .uri="/set_lights",   .method=HTTP_GET,  .handler=set_lights_handler     };
static const httpd_uri_t uri_reprovision    = {
    .uri="/reprovision",  .method=HTTP_GET,  .handler=reprovision_handler    };

static void start_http_server(bool provisioning_mode)
{
    httpd_handle_t server = NULL;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;

    /* Default max_uri_handlers is 8. We now have 7 URIs in normal mode
     * (including /reprovision), so bump to 10 for headroom. */
    config.max_uri_handlers = 10;

    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start HTTP server");
        return;
    }

    if (provisioning_mode) {
        httpd_register_uri_handler(server, &uri_provision_page);
        httpd_register_uri_handler(server, &uri_provision);
        ESP_LOGI(TAG, "Provisioning server ready");
        ESP_LOGI(TAG, "  Open browser -> http://%s", PROV_AP_IP);
        ESP_LOGI(TAG, "  Or POST JSON -> http://%s/provision", PROV_AP_IP);
    } else {
        httpd_register_uri_handler(server, &uri_preview_on);
        httpd_register_uri_handler(server, &uri_light_on);
        httpd_register_uri_handler(server, &uri_all_off);
        httpd_register_uri_handler(server, &uri_set_lights);
        httpd_register_uri_handler(server, &uri_reprovision);   /* NEW */
        ESP_LOGI(TAG, "HTTP server ready (normal mode)");
        ESP_LOGI(TAG, "  To change WiFi: open http://esp32ps.local/reprovision");
    }
}

/* ─── Wi-Fi Event Handler (STA mode) ─────────────────────────────────────── */

static void sta_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();

    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_retry_count++;
        ESP_LOGW(TAG, "Disconnected — retry %d/%d",
                 s_retry_count, MAX_RETRY_BEFORE_PROVISION);

        if (s_retry_count >= MAX_RETRY_BEFORE_PROVISION) {
            /* The saved hotspot has been unreachable for ~45 seconds.
             * Most likely the hotspot name or password changed.
             * Self-reprovision so the user can enter new credentials
             * without needing physical access to the BOOT button. */
            ESP_LOGE(TAG, "====================================================");
            ESP_LOGE(TAG, "  Too many retries. Hotspot appears unreachable.");
            ESP_LOGE(TAG, "  Erasing credentials and entering provisioning.");
            ESP_LOGE(TAG, "  Connect to WiFi: \"%s\"", PROV_AP_SSID);
            ESP_LOGE(TAG, "  Then open:       http://%s", PROV_AP_IP);
            ESP_LOGE(TAG, "====================================================");
            vTaskDelay(pdMS_TO_TICKS(200));
            nvs_erase_creds();
            esp_restart();
        } else {
            esp_wifi_connect();
        }

    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        s_retry_count = 0;   /* Reset counter on successful connection */
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "------------------------------------");
        ESP_LOGI(TAG, "Connected! ESP32 IP : " IPSTR, IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "mDNS hostname       : %s.local", MDNS_HOSTNAME);
        ESP_LOGI(TAG, "Reprovision URL     : http://%s.local/reprovision",
                 MDNS_HOSTNAME);
        ESP_LOGI(TAG, "------------------------------------");
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

/* ─── Wi-Fi STA Init ──────────────────────────────────────────────────────── */

static void wifi_init_sta(const char *ssid, const char *password)
{
    s_wifi_event_group = xEventGroupCreate();

    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &sta_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &sta_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {0};
    strncpy((char *)wifi_config.sta.ssid,     ssid,
            sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char *)wifi_config.sta.password, password,
            sizeof(wifi_config.sta.password) - 1);
    wifi_config.sta.threshold.authmode =
        (strlen(password) == 0) ? WIFI_AUTH_OPEN : WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to \"%s\" (DHCP)...", ssid);
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT,
                        pdFALSE, pdFALSE, portMAX_DELAY);
}

/* ─── Wi-Fi AP Init (provisioning) ───────────────────────────────────────── */

static void wifi_init_ap(void)
{
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t ap_config = {
        .ap = {
            .ssid           = PROV_AP_SSID,
            .ssid_len       = strlen(PROV_AP_SSID),
            .password       = "",
            .max_connection = 4,
            .authmode       = WIFI_AUTH_OPEN,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "====================================================");
    ESP_LOGI(TAG, "  PROVISIONING MODE  (LED 1 blinking)");
    ESP_LOGI(TAG, "  1. Connect phone to Wi-Fi: \"%s\"", PROV_AP_SSID);
    ESP_LOGI(TAG, "  2. Open browser -> http://%s", PROV_AP_IP);
    ESP_LOGI(TAG, "  3. Fill in hotspot name + password, tap Save");
    ESP_LOGI(TAG, "====================================================");
}

/* ─── mDNS ────────────────────────────────────────────────────────────────── */

static void start_mdns(void)
{
    ESP_ERROR_CHECK(mdns_init());
    ESP_ERROR_CHECK(mdns_hostname_set(MDNS_HOSTNAME));
    ESP_ERROR_CHECK(mdns_instance_name_set(MDNS_INSTANCE));
    mdns_service_add(NULL, "_http", "_tcp", 80, NULL, 0);
    ESP_LOGI(TAG, "mDNS: reachable as %s.local", MDNS_HOSTNAME);
}

/* ─── Main ────────────────────────────────────────────────────────────────── */

void app_main(void)
{
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    init_led_gpios();

    /* BOOT button is now a last-resort hardware fallback only.
     * Preferred methods: /reprovision endpoint, or auto-fallback on retries. */
    bool force_provision = boot_button_held();
    if (force_provision) {
        ESP_LOGI(TAG, "BOOT held — clearing credentials, entering provisioning");
        nvs_erase_creds();
    }

    char saved_ssid[64]     = {0};
    char saved_password[64] = {0};
    bool has_creds = (nvs_read_creds(saved_ssid, sizeof(saved_ssid),
                                     saved_password, sizeof(saved_password)) == ESP_OK
                      && strlen(saved_ssid) > 0);

    if (!has_creds || force_provision) {
        /* ── PROVISIONING MODE ─────────────────────────────────────────── */
        wifi_init_ap();
        start_http_server(true);
        /* Blink LED 1 indefinitely. esp_restart() inside provision_handler
         * reboots the chip once credentials are saved. */
        while (1) {
            gpio_set_level(LED_PINS[0], 1);
            vTaskDelay(pdMS_TO_TICKS(500));
            gpio_set_level(LED_PINS[0], 0);
            vTaskDelay(pdMS_TO_TICKS(500));
        }
    } else {
        /* ── NORMAL OPERATION MODE ─────────────────────────────────────── */
        ESP_LOGI(TAG, "Credentials found. Connecting to \"%s\"...", saved_ssid);
        wifi_init_sta(saved_ssid, saved_password);
        start_mdns();
        start_http_server(false);
        /* FreeRTOS scheduler handles everything from here.
         * sta_event_handler will auto-reprovision if the hotspot
         * becomes permanently unreachable. */
    }
}
