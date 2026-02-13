$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr( 'src', e.target.result );
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });
    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            timeout: 60000, // 60 second timeout
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);

                // Get the predicted class and its probability
                var prediction = data.prediction;
                var probability = data.probabilities[prediction];
                var probabilityPercent = (probability * 100).toFixed(2);

                // Display the result
                $('#result').html('Prediction: ' + prediction + '<br>Probability: ' + probabilityPercent + '%');
                console.log('Success!');
            },
            error: function(xhr, status, error) {
                // Hide loader and show error message
                $('.loader').hide();
                $('#btn-predict').show();
                $('#result').fadeIn(600);
                $('#result').html('Error: ' + (xhr.responseJSON ? xhr.responseJSON.error : 'An error occurred during prediction. Please try again.'));
                console.log('AJAX Error:', status, error);
            }
        });
    });

});
