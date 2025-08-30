import sqlite3
import json
from datetime import datetime
import threading
import queue
import logging

class OfflineDataManager:
    """Handle offline data storage and synchronization"""
    
    def __init__(self, db_path='offline_data.db'):
        self.db_path = db_path
        self.sync_queue = queue.Queue()
        self.init_database()
        
        # Start background sync thread
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
    
    def init_database(self):
        """Initialize SQLite database for offline storage"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for offline data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS offline_sos (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                location_lat REAL,
                location_lng REAL,
                request_type TEXT,
                description TEXT,
                created_at TIMESTAMP,
                synced BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS offline_checkins (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                is_safe BOOLEAN,
                location_lat REAL,
                location_lng REAL,
                created_at TIMESTAMP,
                synced BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS offline_aid_requests (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                aid_type TEXT,
                quantity INTEGER,
                description TEXT,
                location_lat REAL,
                location_lng REAL,
                created_at TIMESTAMP,
                synced BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_offline_sos(self, sos_data):
        """Store SOS request offline"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO offline_sos 
            (id, user_id, location_lat, location_lng, request_type, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            sos_data['id'],
            sos_data['user_id'],
            sos_data['location']['lat'],
            sos_data['location']['lng'],
            sos_data['request_type'],
            sos_data['description'],
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
        # Add to sync queue
        self.sync_queue.put(('sos', sos_data['id']))
        
        logging.info(f"SOS request stored offline: {sos_data['id']}")
    
    def store_offline_checkin(self, checkin_data):
        """Store safety check-in offline"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO offline_checkins 
            (id, user_id, is_safe, location_lat, location_lng, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            checkin_data['id'],
            checkin_data['user_id'],
            checkin_data['is_safe'],
            checkin_data['location']['lat'],
            checkin_data['location']['lng'],
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
        self.sync_queue.put(('checkin', checkin_data['id']))
    
    def get_pending_sync_data(self):
        """Get all unsynced data"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unsynced SOS requests
        cursor.execute('SELECT * FROM offline_sos WHERE synced = FALSE')
        unsynced_sos = cursor.fetchall()
        
        # Get unsynced check-ins
        cursor.execute('SELECT * FROM offline_checkins WHERE synced = FALSE')
        unsynced_checkins = cursor.fetchall()
        
        # Get unsynced aid requests
        cursor.execute('SELECT * FROM offline_aid_requests WHERE synced = FALSE')
        unsynced_aid = cursor.fetchall()
        
        conn.close()
        
        return {
            'sos_requests': unsynced_sos,
            'checkins': unsynced_checkins,
            'aid_requests': unsynced_aid
        }
    
    def mark_as_synced(self, data_type, record_id):
        """Mark record as synced"""
        
        table_map = {
            'sos': 'offline_sos',
            'checkin': 'offline_checkins',
            'aid': 'offline_aid_requests'
        }
        
        if data_type not in table_map:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f'UPDATE {table_map[data_type]} SET synced = TRUE WHERE id = ?', (record_id,))
        
        conn.commit()
        conn.close()
        
        logging.info(f"Marked {data_type} record {record_id} as synced")
    
    def _sync_worker(self):
        """Background worker for syncing offline data"""
        
        while True:
            try:
                # Wait for sync requests
                data_type, record_id = self.sync_queue.get(timeout=60)
                
                # Attempt to sync when network is available
                if self._is_network_available():
                    success = self._sync_record(data_type, record_id)
                    if success:
                        self.mark_as_synced(data_type, record_id)
                else:
                    # Put back in queue for later
                    self.sync_queue.put((data_type, record_id))
                
                self.sync_queue.task_done()
                
            except queue.Empty:
                # Periodic sync attempt for old records
                self._sync_all_pending()
            except Exception as e:
                logging.error(f"Sync worker error: {e}")
    
    def _is_network_available(self):
        """Check if network connection is available"""
        try:
            import urllib.request
            urllib.request.urlopen('http://www.google.com', timeout=3)
            return True
        except:
            return False
    
    def _sync_record(self, data_type, record_id):
        """Sync individual record to server"""
        # Implementation would depend on your API endpoints
        # This is a placeholder for the actual sync logic
        try:
            # Get record data
            # Send to server API
            # Return success status
            return True
        except Exception as e:
            logging.error(f"Failed to sync {data_type} record {record_id}: {e}")
            return False
    
    def _sync_all_pending(self):
        """Sync all pending records"""
        if not self._is_network_available():
            return
        
        pending_data = self.get_pending_sync_data()
        
        # Sync each type of data
        for sos_record in pending_data['sos_requests']:
            if self._sync_record('sos', sos_record[0]):
                self.mark_as_synced('sos', sos_record[0])
        
        for checkin_record in pending_data['checkins']:
            if self._sync_record('checkin', checkin_record[0]):
                self.mark_as_synced('checkin', checkin_record[0])
        
        for aid_record in pending_data['aid_requests']:
            if self._sync_record('aid', aid_record[0]):
                self.mark_as_synced('aid', aid_record[0])