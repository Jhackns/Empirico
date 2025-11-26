# 📝 INSTRUCCIONES PARA COMPILAR EN OVERLEAF

## Paso 1: Preparar los archivos

Necesitas tener estos archivos listos para subir:

### Archivos obligatorios:
- ✅ `main.tex` - El artículo completo
- ✅ `bibliografia.bib` - Las referencias (ya lo tienes en user_input_files/)

### Imágenes obligatorias (carpeta results/):
- ✅ `confusion_matrix.png`
- ✅ `roc_curve.png`
- ✅ `probability_distribution.png`

## Paso 2: Crear proyecto en Overleaf

1. Ve a https://www.overleaf.com
2. Si no tienes cuenta, regístrate (es gratis)
3. Click en "New Project" (Nuevo Proyecto)
4. Selecciona "Upload Project" (Subir Proyecto)

## Paso 3: Subir archivos

### Opción A: Subir como ZIP (Recomendado)

Crea una estructura así:
```
proyecto_heladas/
├── main.tex
├── bibliografia.bib
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── probability_distribution.png
```

Comprímelo en un ZIP y súbelo a Overleaf.

### Opción B: Subir archivo por archivo

1. Sube `main.tex`
2. Sube `bibliografia.bib`
3. Crea una carpeta llamada `results` (New Folder)
4. Dentro de `results/`, sube las 3 imágenes PNG

## Paso 4: Configurar el compilador

1. En Overleaf, click en el menú (icono de hamburguesa arriba izquierda)
2. Busca la sección "Settings"
3. En "Compiler", selecciona: **pdfLaTeX**
4. En "Main document", asegúrate que sea: **main.tex**

## Paso 5: Compilar

1. Click en el botón verde "Recompile" o presiona Ctrl+S (Cmd+S en Mac)
2. Espera unos segundos
3. El PDF aparecerá a la derecha

## Paso 6: Resolver problemas comunes

### Error: "File not found: results/confusion_matrix.png"
**Solución**: Asegúrate de que:
- La carpeta se llame exactamente `results` (en minúsculas)
- Los archivos PNG estén dentro de esa carpeta
- Los nombres de archivo sean exactos (con guiones bajos, no espacios)

### Error: "Bibliography not found"
**Solución**: 
- Asegúrate de que `bibliografia.bib` esté en la raíz del proyecto
- Compila 2-3 veces seguidas (BibTeX necesita múltiples pasadas)

### Error: "Package babel Error: Unknown option `spanish'"
**Solución**:
- En el menú de Overleaf → Settings
- Cambia "TeX Live version" a 2023 o superior

### Advertencias de fuentes
**Solución**: Las advertencias sobre fuentes son normales y no afectan el PDF final

## Paso 7: Verificar el PDF

Revisa que el PDF tenga:

✓ Título y autores
✓ Abstract en una columna
✓ Dos columnas en el cuerpo
✓ Todas las secciones (Introducción, Metodología, Resultados, Discusión)
✓ Las 3 figuras se visualicen correctamente
✓ Las 4 tablas con datos
✓ Las referencias al final (numeradas [1], [2], etc.)
✓ Las citas en el texto aparecen como [1], [2], etc.

## Paso 8: Descargar el PDF

1. Click en el icono de descarga (arriba a la derecha)
2. Selecciona "PDF"
3. Guarda el archivo

## 🎨 Personalización opcional

### Cambiar autores

Busca en `main.tex` (línea ~45):
```latex
\author{
\IEEEauthorblockN{Tu Nombre Aquí}
\IEEEauthorblockA{\textit{Tu Departamento} \\
\textit{Tu Universidad}\\
Tu Ciudad, País \\
tu.email@universidad.edu}
}
```

### Cambiar título

Busca en `main.tex` (línea ~42):
```latex
\title{Tu Título Personalizado Aquí}
```

### Modificar abstract

Busca `\begin{abstract}` y edita el contenido

### Agregar más referencias

Edita `bibliografia.bib` siguiendo el formato:
```bibtex
@ARTICLE{ClaveUnica2025,
    author = {Apellido, Nombre},
    title = {Título del Artículo},
    year = {2025},
    journal = {Nombre de la Revista},
    volume = {1},
    pages = {1-10},
    doi = {10.xxxx/xxxxx}
}
```

Luego cita en el texto con: `\cite{ClaveUnica2025}`

## 📊 Verificar que las figuras se vean bien

Las figuras deben aparecer así:

**Figura 1** (Matriz de Confusión):
- Debe mostrar un heatmap azul con números
- Etiquetas en español
- Título: "Matriz de Confusión - Predicción de Heladas"

**Figura 2** (Curva ROC):
- Línea azul ascendente
- Línea punteada diagonal (azar)
- AUC = 0.9999 en la leyenda

**Figura 3** (Distribución de Probabilidades):
- Dos histogramas superpuestos (azul y rojo)
- Etiquetas claras

Si alguna figura no se ve:
1. Verifica que el archivo PNG existe en `results/`
2. Verifica que el nombre sea exacto (sin espacios)
3. Recompila el proyecto

## 🔄 Flujo de compilación completo

Para obtener las referencias correctamente:

1. Primera compilación: pdfLaTeX (genera aux files)
2. Segunda compilación: BibTeX (procesa referencias)
3. Tercera compilación: pdfLaTeX (inserta referencias)
4. Cuarta compilación: pdfLaTeX (resuelve cross-references)

En Overleaf esto es automático si:
- Activas "Auto-compile" en Settings
- O simplemente compilas 2-3 veces manualmente

## 📱 Compartir el proyecto

Para compartir con colaboradores:

1. Click en "Share" (arriba derecha)
2. Invita por email
3. Elige permisos: "Can edit" o "Can view"

Para compartir el link público:
1. Click en "Share"
2. "Turn on link sharing"
3. Copia el link

## 💾 Exportar el código fuente

Si necesitas el código LaTeX para revisión o sumisión:

1. Menu → Download → Source
2. Se descargará un ZIP con todo el proyecto

## 🎓 Para sumisión a conferencia/revista

Cuando estés listo para someter el artículo:

1. Descarga el PDF final
2. Si la revista pide el código fuente, descarga "Source"
3. Si piden separar las figuras, descarga cada PNG individual
4. Revisa las guidelines específicas de la conferencia/revista

## ✅ Checklist antes de someter

- [ ] PDF se compila sin errores
- [ ] Todas las figuras se ven correctamente
- [ ] Todas las tablas tienen datos
- [ ] Las referencias están numeradas correctamente
- [ ] Las citas en el texto coinciden con las referencias
- [ ] Los datos de autores están actualizados
- [ ] El abstract no excede el límite de palabras
- [ ] Cumple con el formato IEEE conference

## 🆘 Soporte

Si tienes problemas en Overleaf:

1. Revisa el "Log" (abajo en el panel de compilación)
2. Lee los errores específicos
3. Busca el error en Google: "latex [tu error]"
4. Contacta el soporte de Overleaf (Help → Contact Us)

## 📚 Recursos adicionales

- Manual de IEEEtran: https://ctan.org/pkg/ieeetran
- Overleaf Documentation: https://www.overleaf.com/learn
- LaTeX Stack Exchange: https://tex.stackexchange.com/

---

**¡Listo! Con estas instrucciones deberías poder compilar el artículo sin problemas.**

Si encuentras algún error específico, copia el mensaje de error y búscalo en Google agregando "latex overleaf" al inicio de tu búsqueda.

¡Éxito con tu artículo! 🎉
