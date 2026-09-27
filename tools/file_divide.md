**[TOOL file_devide]**

**name:** file_devide

**code:** /work#file_divide.sh

**example**
```code
/work#file_divide.sh mylog -s 10000
/work#file_divide.sh mydoc.txt -n 25
```

**description:** разделить файл на части.

параметры: 
* **FILE** файл для обработки
* **-s COUNT** количество строк в фрагменте
* **-b BYTES** количество байт в фрагменте
 
stdout содержит имена созданных частей <FILE_N>

**manager:** true

**[/TOOL]**
