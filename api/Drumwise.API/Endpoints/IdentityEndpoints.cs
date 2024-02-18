using System.Security.Claims;
using Drumwise.Application.Common.Extensions;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Identity;
using Drumwise.Infrastructure.Identity.Constants;
using Microsoft.AspNetCore.Mvc;

namespace Drumwise.API.Endpoints;

public static class IdentityEndpoints
{
    public static void MapAdditionalIdentityEndpoints(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapPost("/customRegister", async Task<IResult> (
            [FromBody] UserRegisterDataRequest userRegisterDataRequest,
            [FromServices] IIdentityService identityService,
            HttpContext context) =>
        {
            var result = await identityService
                .CustomUserRegister(userRegisterDataRequest, context)
                .ConfigureAwait(false);

            return result.ProduceApiResponse();
        });
            
        var manageGroup = endpoints.MapGroup("/manage").RequireAuthorization();
        
        manageGroup.MapPost("/addAdditionalUserData",  async Task<IResult>
                (ClaimsPrincipal claimsPrincipal, [FromBody] AdditionalUserDataRequest additionalUserDataRequest,
                    [FromServices] IIdentityService identityService) =>
            {
                var result = await identityService
                    .RegisterAdditionalUserData(additionalUserDataRequest, claimsPrincipal)
                    .ConfigureAwait(false);

                return result.ProduceApiResponse();
            })
            .Produces(StatusCodes.Status204NoContent)
            .ProducesValidationProblem()
            .ProducesValidationProblem(StatusCodes.Status404NotFound);
        
        
    }
}