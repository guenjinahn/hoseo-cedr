__author__ = 'rainmaker'
# -*- coding: utf-8 -*-

import os
import sys
import time
import signal
import platform
import logging
from logging import handlers
import MySQLdb
from MySQLdb import Error
import smtplib
from datetime import datetime, timedelta
import re
import csv
# Multilanguages
import sys
import threading
reload(sys)
sys.setdefaultencoding('utf-8')

current_dir = os.path.dirname(os.path.realpath(__file__))

g_platform = platform.system()

if g_platform == "Linux":
    LOG_DEFAULT_DIR = '/var/log/dstcs/'
elif g_platform == "Windows":
    LOG_DEFAULT_DIR = '.'
elif g_platform == "Darwin":
    LOG_DEFAULT_DIR = '.'

class ThresholdChecker(object):
    def __init__(self, logger, db_info, intervals):
        # Properties for Daemon
        self.stdin_path = '/dev/null'
        self.pidfile_timeout = 5

        self.db_info = db_info
        self.db = None

        self.logger = logger
        self.is_running = True
        self.intervals = intervals

        self.message = []

    def connect_to_db(self, db_info=None):
        if db_info is None:
            info = self.db_info
        else:
            info = db_info

        try:
            self.db = MySQLdb.connect(
                info['host'],
                info['username'],
                info['password'],
                info['database'])

            self.logger.info('Database connected.')
            self.db.autocommit(True)

            return True

        except Error, e:
            print "2"
            self.logger.error('Database connection failed: %s' % str(e))

            return False

    def disconnect(self):
        if self.db:
            self.db.close()

    def cursor(self):
        if self.db is None:
            return None


        return self.db.cursor(MySQLdb.cursors.DictCursor)


    def process_data(self):
        self.connect_to_db();
        sqls = []
        cursor = None

        try:
            cursor = self.cursor()
            with open('..\\..\\ldapclient\\data.csv', 'rb') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='\n')
                for row in spamreader:
                    s =  " ".join(row)
                    params = str(row).split(',')
                    params_1 = s.split(',')
                    #
                    sql = "INSERT INTO tb_group(pid,group_name_1,group_name_2,group_name_3,group_name_4) VALUES(" + \
                          params_1[0] + ",'" + str(params_1[1]).encode('euc-kr') + "','" + str(params_1[2]).encode('euc-kr')+ "','" + str(
                        params_1[3]).encode('euc-kr') + "','" + str(params_1[4]).encode('euc-kr') + "')";
                    #sql = "INSERT INTO tb_group(pid,group_name_1,group_name_2,group_name_3,group_name_4) VALUES(" + params_1[0] + ",'" + str(params[2]) + "','" + str(params[3]) + "','" + str(params[4]) + "','" + params[5].split('\']')[0] + "')";
                    print(sql)
                    cursor.execute(sql)
            # cursor = self.cursor()
            # sql = "INSERT INTO tb_group(pid,group_name_1,group_name_2,group_name_3,group_name_4) VALUES(1,'aaa','bbb','ccc','ddd')";
            # cursor.execute(sql)

            return []

        except Error, e:
            self.logger.error("Check sql statement failed: %s" % str(e))
            if not self.connect_to_db():
                self.logger.error('Database initialize failed.')
            return []

        finally:
            if cursor is not None:
                cursor.close()

    def run(self):
        self.process_data()

        self.disconnect()

    def init_signal_handler(self):
        def signal_term_handler(signum, frame):
            self.logger.info("Got TERM signal. Stopping application.")
            self.is_running = False

        signal.signal(signal.SIGTERM, signal_term_handler)

if __name__ == '__main__':
    file_logger = logging.getLogger("ThresholdChecker")
    file_logger.setLevel(logging.INFO)

    file_handler = handlers.RotatingFileHandler(
        "th-checker.log",
        maxBytes=(1024 * 1024 * 1),
        backupCount=3
    )

    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    file_logger.addHandler(file_handler)

    app = ThresholdChecker(file_logger, {
        'host': '127.0.0.1',
        'username': 'root',
        'password': '1234',
        'database': 'mondb'
    }, 300)

    if g_platform == "Linux":
    #if False:
        from daemon import runner
        try:
            daemon_runner = runner.DaemonRunner(app)
            daemon_runner.daemon_context.files_preserve = [file_handler.stream]
            daemon_runner.do_action()

        except runner.DaemonRunnerStopFailureError:
            print("Daemon already stopped.")

        except runner.DaemonRunnerStartFailureError:
            print("Daemon start failed.")
    else:
        app.run()



