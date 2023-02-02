#!/usr/bin/env python3

import I2C_LCD_driver
import time
starttime = time.time()
mylcd = I2C_LCD_driver.lcd()
messages = ["It's Jake, Mathafka", "Whoef Whoef       ", "KILL MODE: ACTIVATED", "Where are my testicles?"]
interval = 10
lenmess = len(messages)
i = 0
while True:
    mess = messages[i%lenmess]
    i +=1
    mylcd.lcd_display_string(mess, 1)
    time.sleep(interval - ((time.time() - starttime)%interval))

# Might need to adjust back meter with screwdriver
