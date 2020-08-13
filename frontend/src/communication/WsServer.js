import { server } from WebSocket;

function startServer(){
  var wss = new server({port: 3001});
  wss.on('connection', (ws) =>{
    ws.on('message', (message) => {
      let sendData = {event: 'res', dataa: null};
      message = JSON.parse(message);
      switch(message.event){
        case 'open':
          console.log("receive: %s", message.event);
          break;
        case 'req':
          sendData.data = message.data;
          ws.send(JSON.stringify(sendData));
          break;
        default:
      }
    });
  });
}
