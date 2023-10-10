{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% block methods %}
{% if methods %}
.. rubric:: {{ _('Methods') }}
{% endif %}

.. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}   
   {% endblock %}

{% block attributes %}
{% if attributes %}
.. rubric:: {{ _('Attributes') }}

.. autosummary::
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

.. autoclass:: {{ objname }}
   :members: