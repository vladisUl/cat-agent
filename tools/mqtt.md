**[TOOL mqtt]**

**name:** mqtt

**code:**
```code
mqtt_sub.sh <topic> <field>
mqtt_pub.sh <topic> <payload>
```

**description:**
Доступ к данным, датчикам и устройствам через MQTT.

Для чтения:
```code
/work#mqtt_sub.sh <topic> <field>
```

Для публикации:
```code
/work#mqtt_pub.sh <topic> <payload>
```

* topic — MQTT topic; перечень доступных topic находится ниже
* field — поле из payload
* payload — публикуемое значение в формате, требуемом устройством

Алгоритм чтения:
* выбрать topic по смыслу запроса и описанию топиков
* выбрать нужное field исходя из запроса
* если запрошенного параметра в доступных полях нет, не подменять его другим

**example:**
```code
/work#mqtt_sub.sh zigbee2mqtt/temp_ulica humidity
/work#mqtt_pub.sh zigbee2mqtt/rozetka_komnata state=ON
```

**manager:** true

**[/TOOL]**
