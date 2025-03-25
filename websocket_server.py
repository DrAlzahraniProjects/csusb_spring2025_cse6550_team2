# websocket_server.py
import asyncio
import websockets
import json

async def handler(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        # Simple echo for now - you can enhance this
        response = {"response": data["message"]}
        await websocket.send(json.dumps(response))

start_server = websockets.serve(handler, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
