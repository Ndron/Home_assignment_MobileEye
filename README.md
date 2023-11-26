# How to install and run project  
conda create -n yolov8 python=3.9  
conda activate yolov8  
pip install -r requirements.txt  
python main.py -s run.mp4  

# Assumptions or simplifications.  
1)There is no reconnection to the stream in the script, in case it breaks.
(script was tested on video)  
2)I used conda, preferably use venv or wrap the solution in docker,
 but I didn't spend time on it since it wasn't discussed in the assignment.  
3)The smallest model was taken.  
4)Аll preprocessing, including frame resizing and normalization,
 takes place inside the ultralitix yolov8 library.  
.   
# An explanation of the choices you made in building your pipeline.  
1)All libraries were used according to the instructions, I tried not to use additional ones.  
2)Data is written in jsonl for convenient adding of new rows.  
3)Reading the video stream and model inference placed in one thread to not increase the code size.  
4)The solution is not elegant since I used queues,
 I understand that assigment can be done without them.
 I tried to do it as quickly as possible with my current skils.  
