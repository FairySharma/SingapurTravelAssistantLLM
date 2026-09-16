#!/usr/bin/env python3
"""
MCP Currency Conversion Server for Singapore Travel Assistant
Uses Frankfurter API (free, no API key required, backed by ECB data)
and Open Exchange Rates for broader currency support.
"""

import json
import sys
import urllib.request
from datetime import datetime, timezone


# Currency display names for better UX
CURRENCY_NAMES = {
    "AED": "UAE Dirham",
    "AUD": "Australian Dollar",
    "CAD": "Canadian Dollar",
    "CHF": "Swiss Franc",
    "CNY": "Chinese Yuan",
    "EUR": "Euro",
    "GBP": "British Pound",
    "HKD": "Hong Kong Dollar",
    "IDR": "Indonesian Rupiah",
    "INR": "Indian Rupee",
    "JPY": "Japanese Yen",
    "KRW": "South Korean Won",
    "MYR": "Malaysian Ringgit",
    "NZD": "New Zealand Dollar",
    "PHP": "Philippine Peso",
    "SAR": "Saudi Riyal",
    "SGD": "Singapore Dollar",
    "THB": "Thai Baht",
    "TWD": "Taiwan Dollar",
    "USD": "US Dollar",
    "VND": "Vietnamese Dong",
    "ZAR": "South African Rand",
}


def get_exchange_rate(from_currency: str, to_currency: str) -> dict:
    """
    Fetch exchange rate using Frankfurter API.
    Falls back to Open Exchange Rates API for currencies not covered by Frankfurter.
    """
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()

    # Frankfurter does not support INR, IDR, THB, etc. 
    # We'll use a different free API for broader coverage
    # Using exchangerate-api.com open endpoint (no key, limited but functional)
    
    # Try primary: Frankfurter API (ECB-backed, very reliable for major currencies)
    try:
        url = f"https://api.frankfurter.app/latest?from={from_currency}&to={to_currency}"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        if "rates" in data and to_currency in data["rates"]:
            rate = data["rates"][to_currency]
            return {
                "success": True,
                "from": from_currency,
                "to": to_currency,
                "rate": rate,
                "date": data.get("date", "N/A"),
                "source": "Frankfurter API (European Central Bank data)",
                "source_url": "https://www.frankfurter.app",
            }
    except Exception:
        pass  # Fall through to backup

    # Fallback: Open.er-api.com (free, no key required)
    try:
        url = f"https://open.er-api.com/v6/latest/{from_currency}"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        if data.get("result") == "success" and to_currency in data.get("rates", {}):
            rate = data["rates"][to_currency]
            return {
                "success": True,
                "from": from_currency,
                "to": to_currency,
                "rate": rate,
                "date": data.get("time_last_update_utc", "N/A"),
                "source": "Open Exchange Rates API (open.er-api.com)",
                "source_url": "https://www.exchangerate-api.com",
            }
    except Exception as e:
        pass

    return {
        "success": False,
        "error": f"Could not retrieve exchange rate for {from_currency} to {to_currency}. "
                 "The currency may not be supported or the service is temporarily unavailable.",
    }


def handle_convert_currency(arguments: dict) -> str:
    """Handle currency conversion tool call."""
    amount = arguments.get("amount", 1.0)
    from_currency = arguments.get("from_currency", "USD").upper().strip()
    to_currency = arguments.get("to_currency", "SGD").upper().strip()

    try:
        amount = float(amount)
    except (ValueError, TypeError):
        return "Invalid amount provided. Please provide a numeric value."

    if amount <= 0:
        return "Amount must be greater than zero."

    result = get_exchange_rate(from_currency, to_currency)

    if not result["success"]:
        return f"⚠️ Currency conversion unavailable: {result['error']}"

    rate = result["rate"]
    converted = amount * rate

    from_name = CURRENCY_NAMES.get(from_currency, from_currency)
    to_name = CURRENCY_NAMES.get(to_currency, to_currency)

    # Format numbers nicely
    if converted >= 1000:
        converted_str = f"{converted:,.2f}"
        amount_str = f"{amount:,.2f}"
    else:
        converted_str = f"{converted:.2f}"
        amount_str = f"{amount:.2f}"

    lines = [
        f"💱 **Currency Conversion**",
        f"",
        f"  {amount_str} **{from_currency}** ({from_name})",
        f"  = **{converted_str} {to_currency}** ({to_name})",
        f"",
        f"  Exchange Rate: 1 {from_currency} = {rate:.6f} {to_currency}",
        f"  Rate Date: {result['date']}",
        f"  📡 Source: {result['source']}",
    ]

    # Add travel tips for common conversions
    if to_currency == "SGD":
        # rate = 1 from_currency = X SGD, so inverse_rate = 1 SGD = (1/rate) from_currency
        if rate > 0:
            inv = 1.0 / rate
            lines.append("")
            lines.append("💡 **Singapore Travel Budget Tips (approximate in your currency):**")
            lines.append(f"  • Hawker meal: SGD 4–8 (~{4 * inv:,.0f}–{8 * inv:,.0f} {from_currency})")
            lines.append(f"  • MRT single trip: SGD 1–3 (~{1 * inv:,.0f}–{3 * inv:,.0f} {from_currency})")
            lines.append(f"  • Budget hotel per night: SGD 80–150 (~{80 * inv:,.0f}–{150 * inv:,.0f} {from_currency})")
            lines.append(f"  • Mid-range restaurant meal: SGD 15–40 (~{15 * inv:,.0f}–{40 * inv:,.0f} {from_currency})")

    return "\n".join(lines)


def handle_get_sgd_rates(arguments: dict) -> str:
    """Get current SGD exchange rates against common currencies."""
    currencies_to_check = ["USD", "EUR", "GBP", "INR", "AUD", "JPY", "MYR", "IDR", "THB"]
    
    # Try to get rates
    try:
        url = "https://open.er-api.com/v6/latest/SGD"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        if data.get("result") == "success":
            rates = data.get("rates", {})
            date = data.get("time_last_update_utc", "N/A")
            
            lines = [
                "💱 **Current SGD Exchange Rates**",
                f"📅 Last Updated: {date}",
                f"📡 Source: Open Exchange Rates API (open.er-api.com)",
                "",
                "1 SGD equals:",
            ]
            
            for currency in currencies_to_check:
                if currency in rates:
                    name = CURRENCY_NAMES.get(currency, currency)
                    lines.append(f"  • {rates[currency]:.4f} {currency} ({name})")
            
            return "\n".join(lines)
    except Exception as e:
        pass

    return "⚠️ Unable to retrieve current SGD exchange rates. Please try again later."


# MCP Tool Definitions
TOOLS = [
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another using live exchange rates. Use this when the user wants to convert their travel budget or understand costs in Singapore dollars (SGD). Supported currencies include USD, EUR, GBP, INR, AUD, JPY, SGD, MYR, IDR, THB, HKD, CNY, CAD, CHF, NZD, and more.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "The amount to convert (must be positive).",
                },
                "from_currency": {
                    "type": "string",
                    "description": "The 3-letter ISO currency code to convert FROM (e.g., 'USD', 'INR', 'EUR').",
                },
                "to_currency": {
                    "type": "string",
                    "description": "The 3-letter ISO currency code to convert TO (e.g., 'SGD', 'USD', 'EUR').",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "get_sgd_rates",
        "description": "Get the current exchange rates of Singapore Dollar (SGD) against major currencies including USD, EUR, GBP, INR, AUD, JPY, MYR, IDR, and THB. Use this to show the user a quick overview of SGD rates.",
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
                    "name": "singapore-currency-server",
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
            "convert_currency": handle_convert_currency,
            "get_sgd_rates": handle_get_sgd_rates,
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
                        "content": [{"type": "text", "text": f"Currency conversion error: {str(e)}"}],
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
        return None

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
