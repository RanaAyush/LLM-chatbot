css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
}
.chat-message.user {
    background-color: #128C7E;
    border-top-right-radius: 0;
}
.chat-message.bot {
    background-color: #253342;
    border-top-left-radius: 0;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 51%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
  display:flex;
  align-items:center;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn.dribbble.com/userupload/13167768/file/original-08e29755d8f12fdb9ef53d5b88bfeef0.jpg" alt="opps!">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn.dribbble.com/userupload/5050465/file/original-e97221352fa1248fe66e5b0fd4e5f480.jpg" alt="opps!">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''