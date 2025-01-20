import unittest
from helium import start_chrome, go_to, kill_browser
import logging
import os

# 配置日志
LOG_FILE = "simple_test.log"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TestSimple(unittest.TestCase):

    def setUp(self):
        logging.info("Starting test case")
        self.driver = start_chrome()  # 启动 Chrome 浏览器

    def tearDown(self):
        logging.info("Finishing test case")
        kill_browser()  # 关闭浏览器

    def test_google_title(self):
        logging.info("Starting test_google_title")
        go_to("https://www.google.com")  # 访问 Google 首页
        title = self.driver.title # 获取页面标题

        # 断言：检查页面标题是否包含 "Google"
        self.assertIn("Google", title)
        logging.info("Finished test_google_title")

if __name__ == "__main__":
    unittest.main()