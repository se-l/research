# greeting.jl

using HTTP

"""
    greeting_handler()

Handles the root route (`/`), returning an HTML page with WebSocket chat UI.
"""
function greeting_handler(req::HTTP.Messages.Request)
    pid = getpid()

    # Optional: extract client info from request if needed
    # client_id = get(Genie.Requests.ip"X-Client-Id", "unknown")
    client_id = "unknown"
    @info "Connected $pid"
    body = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Chat</title>
        </head>
        <body>
            <h2>
            PID $pid. Your Client ID is $client_id</br>
            WebSocket Chat
            </h2>
            <form action="" onsubmit="sendMessage(event)">
                <input type="text" id="messageText" autocomplete="off"/>
                <button>Send</button>
            </form>
            <ul id='messages'>
            </ul>
            <script>
                var ws = new WebSocket("ws://127.0.0.1:8002/ws");
                ws.onmessage = function(event) {
                    var messages = document.getElementById('messages')
                    var message = document.createElement('li')
                    var content = document.createTextNode(event.data)
                    message.appendChild(content)
                    messages.appendChild(message)
                };
                function sendMessage(event) {
                    var input = document.getElementById("messageText")
                    ws.send(input.value)
                    input.value = ''
                    event.preventDefault()
                }
            </script>
        </body>
    </html>
    """

    HTTP.Response(200, ["Content-Type" => "text/html"], body)
end