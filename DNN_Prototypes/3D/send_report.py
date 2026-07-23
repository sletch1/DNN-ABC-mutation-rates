"""Email a plaintext report from stat86 to the user.

Reuses the only outbound path that works on stat86: relay straight to Google's MX
for vt.edu (the local Postfix MTA is down). No credentials needed -- stat86's IP is
allowed to relay to vt.edu. A Message-ID header is required or Google rejects.

Exposes send(subject, text). CLI sends a plaintext body from a file or string.
"""
import argparse
import ssl
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid, formatdate

TO_ADDR = "sach@vt.edu"
FROM_ADDR = "sach@stat86.stat.vt.edu"
MX = "alt1.aspmx.l.google.com"


def send(subject, text):
    m = EmailMessage()
    m["From"] = FROM_ADDR
    m["To"] = TO_ADDR
    m["Subject"] = subject
    m["Message-ID"] = make_msgid(domain="stat86.stat.vt.edu")
    m["Date"] = formatdate(localtime=True)
    m.set_content(text)
    s = smtplib.SMTP(MX, 25, timeout=40)
    try:
        s.ehlo("stat86.stat.vt.edu")
        s.starttls(context=ssl.create_default_context())
        s.ehlo("stat86.stat.vt.edu")
        s.send_message(m)
    finally:
        try:
            s.quit()
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--body")
    ap.add_argument("--body-file")
    args = ap.parse_args()
    body = args.body or ""
    if args.body_file:
        with open(args.body_file) as f:
            body = f.read()
    send(args.subject, body)
    print("email sent to", TO_ADDR)
