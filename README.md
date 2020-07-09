# Face_recognition_using_HaarCascade
Face recognition using HaarCascade and OpenCV


### 실행 순서
1. face_detection.py : id 입력 후 카메라와 haar feacture를 통해 얼굴을 detection, 저장  
2. face_training.py : directory의 id file들이 label. 학습  
3. face_recognition.py : 학습된 파일을 활용해 카메라를 통한 얼굴인식.  


### 참고사항  
face_traingin.py  
~~~   
names = ['', '', '']   
~~~
학습시 원하는 사람의 이름 입력. 
