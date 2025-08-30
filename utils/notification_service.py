
import requests
import json
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

class NotificationService:
    """Handle various notification channels for SAHAYAK+"""
    
    def __init__(self, config):
        self.config = config
        self.twilio_client = None
        
        # Initialize Twilio if configured
        if config.get('TWILIO_SID') and config.get('TWILIO_TOKEN'):
            self.twilio_client = Client(
                config['TWILIO_SID'], 
                config['TWILIO_TOKEN']
            )
    
    def send_sms_alert(self, phone_number, message, priority='normal'):
        """Send SMS alert via Twilio"""
        
        if not self.twilio_client:
            logging.error("Twilio not configured")
            return False
        
        try:
            message = self.twilio_client.messages.create(
                body=message,
                from_=self.config['TWILIO_PHONE'],
                to=phone_number
            )
            
            logging.info(f"SMS sent to {phone_number}: {message.sid}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send SMS: {e}")
            return False
    
    def send_push_notification(self, user_tokens, title, body, data=None):
        """Send push notification via Firebase"""
        
        fcm_url = "https://fcm.googleapis.com/fcm/send"
        headers = {
            'Authorization': f"key={self.config['FCM_SERVER_KEY']}",
            'Content-Type': 'application/json'
        }
        
        notification_data = {
            'registration_ids': user_tokens,
            'notification': {
                'title': title,
                'body': body,
                'sound': 'default',
                'priority': 'high'
            }
        }
        
        if data:
            notification_data['data'] = data
        
        try:
            response = requests.post(fcm_url, headers=headers, json=notification_data)
            
            if response.status_code == 200:
                logging.info(f"Push notification sent to {len(user_tokens)} devices")
                return True
            else:
                logging.error(f"Failed to send push notification: {response.text}")
                return False
                
        except Exception as e:
            logging.error(f"Push notification error: {e}")
            return False
    
    def send_email_alert(self, recipients, subject, message, html_content=None):
        """Send email alert"""
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config['EMAIL_FROM']
            msg['To'] = ', '.join(recipients)
            
            # Add text content
            text_part = MIMEText(message, 'plain')
            msg.attach(text_part)
            
            # Add HTML content if provided
            if html_content:
                html_part = MIMEText(html_content, 'html')
                msg.attach(html_part)
            
            # Send email
            server = smtplib.SMTP(self.config['SMTP_HOST'], self.config['SMTP_PORT'])
            server.starttls()
            server.login(self.config['EMAIL_USER'], self.config['EMAIL_PASS'])
            server.send_message(msg)
            server.quit()
            
            logging.info(f"Email sent to {recipients}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email: {e}")
            return False
    
    def broadcast_emergency_alert(self, alert_data):
        """Broadcast emergency alert through multiple channels"""
        
        results = {}
        
        # SMS alerts to nearby rescuers
        if alert_data.get('nearby_rescuers'):
            sms_message = f"EMERGENCY ALERT: {alert_data['type']} at {alert_data['location']}. Priority: {alert_data['priority']}. Respond immediately."
            
            for rescuer in alert_data['nearby_rescuers']:
                if rescuer.get('phone'):
                    results[f"sms_{rescuer['id']}"] = self.send_sms_alert(
                        rescuer['phone'], sms_message, 'high'
                    )
        
        # Push notifications to rescuer apps
        if alert_data.get('rescuer_tokens'):
            results['push_notification'] = self.send_push_notification(
                alert_data['rescuer_tokens'],
                f"🚨 {alert_data['priority'].upper()} EMERGENCY",
                f"{alert_data['type']} emergency at {alert_data['location']}",
                {
                    'type': 'sos_alert',
                    'sos_id': alert_data['sos_id'],
                    'priority': alert_data['priority']
                }
            )
        
        # Email to coordination centers
        if alert_data.get('admin_emails'):
            email_subject = f"SAHAYAK+ Emergency Alert - {alert_data['priority'].upper()} Priority"
            email_body = f"""
            New emergency request received:
            
            Type: {alert_data['type']}
            Location: {alert_data['location']}
            Priority: {alert_data['priority']}
            Survivor: {alert_data.get('survivor_name', 'Unknown')}
            Description: {alert_data.get('description', 'No description provided')}
            
            Time: {alert_data['timestamp']}
            SOS ID: {alert_data['sos_id']}
            
            Please coordinate rescue efforts immediately.
            """
            
            results['email_alert'] = self.send_email_alert(
                alert_data['admin_emails'],
                email_subject,
                email_body
            )
        
        return results