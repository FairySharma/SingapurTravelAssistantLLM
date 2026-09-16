#!/usr/bin/env python3
"""
MCP Weather Server for Singapore Travel Assistant
Uses Open-Meteo API (free, no API key required)
Provides current weather and forecasts for Singapore
"""

import json
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone

# Singapore coordinates
SINGAPORE_LAT = 1.3521
SINGAPORE_LON = 103.8198
SINGAPORE_TIMEZONE = "Asia/Singapore"


def get_weather_description(wmo_code: int) -> str:
    """Convert WMO weather code to human-readable description."""
    codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return codes.get(wmo_code, f"Unknown ({wmo_code})")


def fetch_weather() -> dict:
    """Fetch current weather and 7-day forecast from Open-Meteo API."""
    params = {
        "latitude": SINGAPORE_LAT,
        "longitude": SINGAPORE_LON,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
            "uv_index",
        ],
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "uv_index_max",
        ],
        "timezone": SINGAPORE_TIMEZONE,
        "forecast_days": 7,
    }

    # Build query string
    query_parts = []
    for key, value in params.items():
        if isinstance(value, list):
            query_parts.append(f"{key}={','.join(value)}")
        else:
            query_parts.append(f"{key}={value}")

    url = "https://api.open-meteo.com/v1/forecast?" + "&".join(query_parts)

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data
    except Exception as e:
        return {"error": str(e)}


def format_weather_response(data: dict, query_type: str = "current") -> str:
    """Format the weather API response into a readable string."""
    if "error" in data:
        return f"Weather data unavailable: {data['error']}"

    location = "Singapore"
    lines = [f"📍 Weather for {location}"]
    lines.append(f"🕐 Data retrieved: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"📡 Source: Open-Meteo API (open-meteo.com)")
    lines.append("")

    if query_type in ("current", "both") and "current" in data:
        c = data["current"]
        wmo = c.get("weather_code", 0)
        desc = get_weather_description(wmo)
        temp = c.get("temperature_2m", "N/A")
        feels = c.get("apparent_temperature", "N/A")
        humidity = c.get("relative_humidity_2m", "N/A")
        precip = c.get("precipitation", 0)
        wind = c.get("wind_speed_10m", "N/A")
        uv = c.get("uv_index", "N/A")

        lines.append("🌤️ **Current Conditions:**")
        lines.append(f"  • Condition: {desc}")
        lines.append(f"  • Temperature: {temp}°C (feels like {feels}°C)")
        lines.append(f"  • Humidity: {humidity}%")
        lines.append(f"  • Precipitation: {precip}mm")
        lines.append(f"  • Wind Speed: {wind} km/h")
        lines.append(f"  • UV Index: {uv}")
        lines.append("")

        # Activity recommendation
        is_rainy = wmo in [51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99]
        is_sunny = wmo in [0, 1, 2]
        if is_rainy or precip > 0.5:
            lines.append("☔ **Activity Advice:** Rain detected — consider indoor attractions like Gardens by the Bay domes, ArtScience Museum, or shopping malls.")
        elif is_sunny:
            lines.append("☀️ **Activity Advice:** Great weather for outdoor activities — Merlion Park, Botanic Gardens, Southern Ridges walk, or East Coast Park.")
        else:
            lines.append("🌤️ **Activity Advice:** Partly cloudy conditions — most outdoor activities are fine. Carry an umbrella as Singapore weather can change quickly.")

    if query_type in ("forecast", "both") and "daily" in data:
        lines.append("")
        lines.append("📅 **7-Day Forecast:**")
        daily = data["daily"]
        dates = daily.get("time", [])
        codes = daily.get("weather_code", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        precip_sums = daily.get("precipitation_sum", [])
        precip_probs = daily.get("precipitation_probability_max", [])
        uv_maxes = daily.get("uv_index_max", [])

        for i, date in enumerate(dates):
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                day_name = date_obj.strftime("%A, %d %b")
            except Exception:
                day_name = date

            code = codes[i] if i < len(codes) else 0
            desc = get_weather_description(code)
            max_t = max_temps[i] if i < len(max_temps) else "N/A"
            min_t = min_temps[i] if i < len(min_temps) else "N/A"
            precip = precip_sums[i] if i < len(precip_sums) else 0
            prob = precip_probs[i] if i < len(precip_probs) else 0
            uv = uv_maxes[i] if i < len(uv_maxes) else "N/A"

            rain_icon = "☔" if (prob and prob > 50) or (precip and precip > 2) else "☀️" if code in [0, 1, 2] else "🌤️"
            lines.append(f"  {rain_icon} **{day_name}:** {desc}, {min_t}–{max_t}°C, Rain: {precip}mm ({prob}% chance), UV: {uv}")

    return "\n".join(lines)


def handle_get_current_weather(arguments: dict) -> str:
    """Handle get_current_weather tool call."""
    data = fetch_weather()
    return format_weather_response(data, "current")


def handle_get_weather_forecast(arguments: dict) -> str:
    """Handle get_weather_forecast tool call."""
    days = arguments.get("days", 3)
    data = fetch_weather()
    if "daily" in data:
        # Limit to requested number of days
        for key in data["daily"]:
            data["daily"][key] = data["daily"][key][:days]
    return format_weather_response(data, "forecast")


def handle_get_full_weather(arguments: dict) -> str:
    """Handle get_full_weather tool call (current + forecast)."""
    data = fetch_weather()
    return format_weather_response(data, "both")


# MCP Tool Definitions
TOOLS = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather conditions in Singapore including temperature, humidity, precipitation, wind speed, and UV index. Use this when the user asks about current weather.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_weather_forecast",
        "description": "Get the weather forecast for Singapore for the next 1–7 days. Returns daily high/low temperatures, weather condition, precipitation amount, precipitation probability, and UV index. Use this when the user asks about future weather or trip planning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of forecast days (1–7). Default is 3.",
                    "minimum": 1,
                    "maximum": 7,
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_full_weather",
        "description": "Get both current weather conditions and the 7-day forecast for Singapore in one response. Use this when the user needs comprehensive weather information for trip planning.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def handle_request(request: dict) -> dict:
    """Handle an incoming JSON-RPC request."""
    method = request.get("method")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "singapore-weather-server",
                    "version": "1.0.0",
                },
            },
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        handlers = {
            "get_current_weather": handle_get_current_weather,
            "get_weather_forecast": handle_get_weather_forecast,
            "get_full_weather": handle_get_full_weather,
        }

        if tool_name in handlers:
            try:
                result = handlers[tool_name](arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": result}],
                        "isError": False,
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error fetching weather: {str(e)}"}],
                        "isError": True,
                    },
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

    elif method == "notifications/initialized":
        return None  # No response needed for notifications

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def main():
    """Main stdio loop for MCP server."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            request = json.loads(line)
            response = handle_request(request)

            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Server error: {str(e)}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
