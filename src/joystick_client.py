import pygame
import os
import time
import requests
import json

axis_1, axis_0, axis_3 = 0.0, 0.0, 0.0
button_x, button_b = False, False
joystick_opened = False

def joystick_client():
    global axis_1, axis_0, axis_3, button_x, button_b, joystick_opened

    joystick_use = True
    pygame.init()
    try:
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
        print(f"手柄已连接: {joystick.get_name()}")
        print("按键映射:")
        print("  X button (button 3): 重置环境")
        print("  B button (button 1): 切换录制状态")
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

            # read button states
            # X button (button 3): reset environment
            # B button (button 1): toggle recording
            button_x = joystick.get_button(3)
            button_b = joystick.get_button(1)

            print(f"axes: {axis_1:.3f}, {axis_0:.3f}, {axis_3:.3f} | buttons: X={button_x}, B={button_b}")

            # 发送到 env_human_ee.py 服务端 (FastAPI joystick server on port 8081)
            url = "http://127.0.0.1:8081/joystick"
            data = {
                "axis_0": axis_0,
                "axis_1": axis_1,
                "axis_3": axis_3,
                "button_x": button_x,
                "button_b": button_b
            }
            try:
                response = requests.post(url, json=data, timeout=1)
                print(f"Sent: {data}, Response: {response.text}")
            except Exception as e:
                print(f"Error sending: {e}")

            pygame.time.delay(100)

if __name__ == '__main__':
    joystick_client()
