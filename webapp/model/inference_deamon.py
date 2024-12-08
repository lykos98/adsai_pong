import schedule
import time
import sys
import threading

sys.path.append(".")
from main import update

def run_update():
    print('updating predictions...')
    did_something = update()
    if did_something:
        print('updated predictions')
        
# Schedule the task
schedule.every().hour.do(run_update)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    # Run the update once at the beginning
    update()
    
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    # Keep the main thread alive
    while True:
        time.sleep(1)