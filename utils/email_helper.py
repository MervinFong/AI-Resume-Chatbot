import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Email Credentials (from .env) ---
SENDER_EMAIL = os.getenv("mervinfkn@gmail.com")  # Example: your_email@gmail.com
SENDER_PASSWORD = os.getenv("yleq wehy unqy armk")  # Gmail App Password

# --- Function to Send Email (Plain or HTML) ---
def send_email(to_email, subject, body, is_html=False):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    try:
        # Setup MIME
        message = MIMEMultipart()
        message['From'] = SENDER_EMAIL
        message['To'] = to_email
        message['Subject'] = subject

        if is_html:
            message.attach(MIMEText(body, 'html'))
        else:
            message.attach(MIMEText(body, 'plain'))

        # Connect and Send
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, message.as_string())
        server.quit()

        print(f"✅ Email sent successfully to {to_email}")

    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")
