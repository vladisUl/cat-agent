**[TOOL mqtt]**

**name:** mqtt

**code:**
```code
mqtt_sub.sh
mqtt_pub.sh
```

**description:**
* Доступ к данным, датчикам и устройствам через MQTT
* Чтения данных mqtt
```code
mqtt_sub.sh <topic> <field>
```

Публикация данных mqtt
```code
mqtt_pub.sh <topic> <field>
```

* topic - это topic mqtt, перечень доступных topic находится ниже
* field - это поля доступные в payload

Алгоритм:
* Выбрать топик по смыслу запроса и описанию топиков
* Выбрать нужное поле (поля) исходя из запроса

**Example:**
```code
/work#mqtt_sub.sh zigbee2mqtt/temp_ulica humidity

/work#mqtt_pub.sh zigbee2mqtt/rozetka state=ON
```
**manager:** true

**[/TOOL]**
