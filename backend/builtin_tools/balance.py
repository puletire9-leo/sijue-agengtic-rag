"""DeepSeek API 余额查询工具."""
import os

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


@tool("check_api_balance")
def check_api_balance() -> str:
    """Query the DeepSeek API account balance.

    Use this when the user asks:
    - "How much API credit do I have left?"
    - "Check my balance"
    - "余额多少" / "查余额" / "API还剩多少钱"

    Returns:
        Account balance in CNY/USD, including total, granted, and topped-up amounts.
    """
    import requests

    api_key = os.getenv("ARK_API_KEY")
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com")

    try:
        resp = requests.get(
            f"{base_url}/user/balance",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return f"查询余额失败: HTTP {resp.status_code}"

        data = resp.json()
        if not data.get("is_available"):
            return "账户余额不可用，无法调用 API。"

        lines = ["DeepSeek API 账户余额："]
        for info in data.get("balance_infos", []):
            currency = info.get("currency", "?")
            total = info.get("total_balance", "0")
            granted = info.get("granted_balance", "0")
            topped_up = info.get("topped_up_balance", "0")
            lines.append(f"- 总余额: {total} {currency}")
            lines.append(f"  充值余额: {topped_up} {currency}")
            lines.append(f"  赠金余额: {granted} {currency}")

        return "\n".join(lines)
    except Exception as e:
        return f"查询余额异常: {e}"
