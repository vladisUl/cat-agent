**[TOOL file_divide]**

**name:** file_divide

**code:**
```code
/work#file_divide.sh
```

**description:** разделить текстовый DATA-файл на части без разрыва строк.

**параметры:**
* **FILE** — логический путь к файлу внутри DATA
* **-s BYTES** — целевой размер части в байтах; граница переносится вперёд до конца текущей строки
* **-n COUNT** — количество частей

**example:**
```code
/work#file_divide.sh mylog -s 10000
/work#file_divide.sh mydoc.txt -n 25
```

stdout содержит логические имена созданных частей 1FILE, 2FILE ...

**manager:** true

**[/TOOL]**
