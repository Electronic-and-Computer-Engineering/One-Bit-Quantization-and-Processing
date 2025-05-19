import smtplib
import ssl
from email.message import EmailMessage
import os

def send_gmail(sender, recipient, subject, body, app_password=None, attachments=None):
    """
    Sends an email using Gmail with optional attachments (via STARTTLS).

    Parameters:
    - sender (str): Gmail address (e.g. 'you@gmail.com')
    - recipient (str or list): Receiver(s)
    - subject (str): Email subject
    - body (str): Plaintext body
    - app_password (str): Gmail App password or from env var 'GMAIL_PASS'
    - attachments (list of file paths): Optional list of file paths to attach
    """
    #if app_password is None:
    #    app_password = os.getenv('GMHANS')
    #if not app_password:
    #    raise ValueError("No app password provided and 'GMAIL_PASS' environment variable not set.")

    msg = EmailMessage()
    msg['From'] = sender
    msg['To'] = recipient if isinstance(recipient, str) else ', '.join(recipient)
    msg['Subject'] = subject
    msg.set_content(body)

    if attachments:
        for path in attachments:
            try:
                with open(path, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(path)
                msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
            except Exception as e:
                print(f"⚠ Fehler beim Anhängen von '{path}': {e}")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls(context=context)
            smtp.login(sender, 'skkicefkgvwdyzjf')
            smtp.send_message(msg)
            print(f"✔ E-Mail erfolgreich an {recipient} gesendet.")
    except Exception as e:
        print(f"❌ Fehler beim Senden der E-Mail: {e}")

__all__ = ['send_gmail']
