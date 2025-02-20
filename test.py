import pytest
from datetime import datetime

if __name__ == "__main__":
    # 获取当前时间并格式化
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 生成HTML测试报告，并启用详细输出
    pytest.main(["-v", f"--html=test_reports/report_{time}.html", "test"])