import pygame
import os
import time
import requests
import json

axis_1, axis_0, axis_3 = 0.0, 0.0, 0.0
joystick_opened = False

def joystick_client():
    global axis_1, axis_0, axis_3, joystick_opened

    joystick_use = True
    pygame.init()
    try:
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")
        return

    if joystick_opened and joystick_use:
        while True:
            # get joystick input in main thread (required for macOS)
            pygame.event.get()

            # update robot command
            axis_1 = -joystick.get_axis(1) * 1
            axis_0 = -joystick.get_axis(0) * 1
            axis_3 = -joystick.get_axis(3) * 1

            print(axis_1, axis_0, axis_3)

            # 发送到 env_human_ee.py 服务端 (FastAPI joystick server on port 8081)
            url = "http://127.0.0.1:8081/joystick"
            data = {
                "axis_0": axis_0,
                "axis_1": axis_1,
                "axis_3": axis_3
            }
            try:
                response = requests.post(url, json=data, timeout=1)
                print(f"Sent: {data}, Response: {response.text}")
            except Exception as e:
                print(f"Error sending: {e}")

            pygame.time.delay(100)

if __name__ == '__main__':
    joystick_client()
