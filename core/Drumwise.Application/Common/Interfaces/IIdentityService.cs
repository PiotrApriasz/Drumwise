using System.Security.Claims;
using Drumwise.Application.Common.Models;
using Drumwise.Application.Common.Models.Identity;
using Microsoft.AspNetCore.Http;

namespace Drumwise.Application.Common.Interfaces;

public interface IIdentityService
{
    Task<string?> GetUserNameAsync(string userId);

    Task<string?> GetUserFullNameIfAvailable(string userId);

    Task<Result> CustomUserRegister(UserRegisterDataRequest userRegisterDataRequest,
        HttpContext context);

    Task<Result> RegisterAdditionalUserData(AdditionalUserDataRequest additionalUserDataRequest,
        ClaimsPrincipal claimsPrincipal);

    Task<Result> DeleteUserAsync(string userId);
}