$(document).ready(function () {
  // <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
  var movieData = JSON.parse(sessionStorage.getItem('movieData'));
  console.log(movieData.image);
      // Movie Information
      $('#name').val(movieData.name);
      $('#price').val(movieData.price);
      $('#date').val(movieData.date);
      $('#venue').val(movieData.venue);
      $('#image').attr("src",movieData.image);
    $('#update').on('click', function () {
          var formData = new FormData();
          formData.append('name', `${$('#name').val()}`);
          formData.append('price', `${$('#price').val()}`);
          formData.append('date', `${$('#date').val()}`);
          formData.append('venue', `${$('#venue').val()}`);
          formData.append('image', `${$('#image').attr("src")}`);
          var fileInput = $('#formFile')[0]; // 获取文件输入元素
          if (fileInput.files.length > 0) {
            formData.append('profileImage', fileInput.files[0]); // 添加图片文件到 FormData
          }
          $.ajax({
            url: '/admin/create',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
              if (data.status === 'success') {
                alert(`Update successfully}!`);
                window.open('/admin_show_movie.html', '_self');
              } else {
                alert(data.message);
              }
            },
            error: function (xhr) {
              const response = xhr.responseJSON || JSON.parse(xhr.responseText);
              if (response && response.message) {
                alert(response.message);
              } else {
                alert('Unknown error');
              }
              console.error('Error:', response);
            },
          });
      });
});
