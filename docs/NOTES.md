TENIAMO LA VELOCITA' A 1

Q0 - Q1
- O si aggiorna facendo delle stime
- O si calcola alla fine


Potrebbe essere un problema il fatto che q0 sia prima di uno switch mentre q1 sia dopo
- obs = curr, next_obs = roll all until next switch, (puo essere poco realistica)

    - si potrebbe mettere esperienza non solo agli switch ma sempre
    - oppure fare un guess in forward delle mappe
        - alternativamente al prossimo switch aggiungere l'esperienza passata
            - la reward diventa la reward fra q0 e q1
            - bisogna detectare i crash e dare una reward molto negativa
                - sempre
                - in caso di deadlock bisogna detectare e dare reward a fine ep
            - aggiungo esperienza se sta fermo q0 == q1

- oppure dare alla rete action / q0 e anche la speed

BACKLOG
- edit rewards

GUARDARE
- buffer con priorita
- backpropagation su anche su q1?
- dqn una fa calcolo l'altra l'update


ASPERTORARI
FREE (Giovedi Pome / Venerdi Pome)
- Lunedi Pomeriggio
- Martedi Pomeriggio
- Mercoledi Pomeriggio
- Venerdi Mattina

APRILE:
ALLDAYS