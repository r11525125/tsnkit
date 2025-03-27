import asyncio
import websockets

async def connect_to_websocket():
    uri = "ws://169.254.158.164:8080/network/ping"
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send("Hello, WebSocket!")
            while True:
                try:
                    response = await websocket.recv()
                    print("Received:", response)
                except websockets.exceptions.ConnectionClosedOK:
                    print("WebSocket connection closed.")
                    break
    except websockets.exceptions.WebSocketException as e:
        print("WebSocket error:", e)

async def main():
    await connect_to_websocket()

asyncio.run(main())
