import smtplib
import pyttsx3
from email.message import EmailMessage
import os

EMAIL_ADDRESS = os.getenv("EMAIL_USER", "your@email.com").strip().replace('"', '').replace("'", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASS", "yourpassword").strip().replace('"', '').replace("'", "")

def send_email(to, subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)
    print(f"✅ Email sent to {to}")

def speak_message(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

from twilio.rest import Client

def make_phone_call(to_number: str, message: str):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_phone = os.getenv("TWILIO_PHONE")

    client = Client(account_sid, auth_token)
    
    call = client.calls.create(
        twiml=f'<Response><Say>{message}</Say></Response>',
        to=to_number,
        from_=twilio_phone
    )
    print(f"📞 Call initiated: {call.sid}")
