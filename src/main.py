# import
import detector
import os

# reduce logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# run
detector.main()
