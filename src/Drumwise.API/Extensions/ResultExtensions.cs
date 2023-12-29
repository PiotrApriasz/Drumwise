using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Exceptions;
using Drumwise.Application.Common.Models;
using Microsoft.AspNetCore.Mvc;

namespace Drumwise.Application.Common.Extensions;

public static class ResultExtensions
{
    public static T Match<T>(this Result result, Func<T> onSuccess, Func<T> onFailure)
    {
        return result.IsSuccess ? onSuccess() : onFailure();
    }

    public static Func<IResult> ProduceErrorApiResponse(this Result result)
    {
        return result.ResultType switch
        {
            ResultType.BadRequest => () =>
                TypedResults.BadRequest(CreateProblemResponse(result.Errors,
                    StatusCodes.Status400BadRequest)),
            ResultType.NotFound => () => TypedResults.NotFound(CreateProblemResponse(result.Errors,
                StatusCodes.Status404NotFound)),
            _ => () => TypedResults.Problem("Unsupported result type value", 
                statusCode: StatusCodes.Status500InternalServerError)
        };
    }

    public static Func<IResult> ProduceSuccessApiResponse(this Result result, object? resultData = null)
    {
        return result.ResultType switch
        {
            ResultType.Ok => () => TypedResults.Ok(resultData),
            ResultType.NoContent => TypedResults.NoContent,
            _ => () => TypedResults.Problem("Unsupported result type value", 
                statusCode: StatusCodes.Status500InternalServerError)
        };
    }

    private static ValidationProblemDetails CreateProblemResponse(IEnumerable<Error> errors, 
        int statusCode)
    {
        var validationProblemDetails = new ValidationProblemDetails()
        {
            Type = GetErrorType(statusCode),
            Status = statusCode
        };

        foreach (var error in errors)
        {
            validationProblemDetails.Errors.Add(error.Code, [error.Description]);
        }

        return validationProblemDetails;
    }

    private static string GetErrorType(int statusCode)
    {
        return statusCode switch
        {
            StatusCodes.Status400BadRequest => "https://tools.ietf.org/html/rfc9110#section-15.5.1",
            StatusCodes.Status404NotFound => "https://tools.ietf.org/html/rfc9110#section-15.5.5",
            _ => "Unknown status code type"
        };
    }
}