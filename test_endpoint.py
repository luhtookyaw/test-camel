from camel_agent import CamelCounselingSession

sess = CamelCounselingSession(
    vllm_server="https://vllm-camel-api.politestone-cb09a6ac.eastus.azurecontainerapps.io/v1",
    model_id="LangAGI-Lab/camel",
)

intake = sess.build_intake_form(
    name="Laura",
    age="45",
    gender="female",
    occupation="Office Job",
    education="College Graduate",
    marital_status="Single",
    family_details="Lives alone",
)

reason = "I feel anxious and overwhelmed at work and can't sleep."

sess.start(intake_form=intake, reason=reason, first_client_message="I'm stressed and keep overthinking.")
print("CBT plan:\n", sess.cbt_plan)

reply = sess.step("I keep thinking I'll mess everything up.")
print("Counselor:", reply)
